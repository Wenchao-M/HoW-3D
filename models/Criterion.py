import torch
import torch.nn.functional as F
from torch import nn
from utils.utils import is_dist_avail_and_initialized,get_world_size


class SetCriterion(nn.Module):

    def __init__(self, num_class, weight_dict, eos_coef, losses,label_loss_params, label_loss_func='cross_entropy',matcher=None):

        super().__init__()
        self.num_classes = num_class
        self.matcher = matcher
        self.label_loss_func = label_loss_func
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.label_loss_params = label_loss_params

    def loss_junctions_labels(self, outputs, targets, num_items, log=False, origin_indices=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_lines]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(origin_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, origin_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.label_loss_func == 'cross_entropy':
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        elif self.label_loss_func == 'focal_loss':
            loss_ce = self.label_focal_loss(src_logits.transpose(1, 2), target_classes, self.empty_weight,
                                            **self.label_loss_params)
        else:
            raise ValueError()

        losses = {'loss_ce': loss_ce}
        return losses


    def loss_junctions(self, outputs, targets, num_items, origin_indices=None):
        assert 'pred_juncs' in outputs

        idx = self._get_src_permutation_idx(origin_indices)
        src_lines = outputs['pred_juncs'][idx]
        target_lines = torch.cat([t['junction_hidden'][i] for t, (_, i) in zip(targets, origin_indices)], dim=0).reshape((-1,2))

        loss_line = F.l1_loss(src_lines, target_lines, reduction='none')
        losses = {}
        losses['loss_junc'] = loss_line.sum() / num_items

        return losses

    def loss_depth(self, outputs, targets,num_items,origin_indices=None):
        assert 'pred_depth' in outputs
        idx = self._get_src_permutation_idx(origin_indices)
        scr_depth = outputs['pred_depth'][idx]
        target_depth = torch.cat([t['juncs_hidden_3D'][i,-1] for t, (_, i) in zip(targets, origin_indices)],dim=0).reshape((-1, 1))

        loss_depth = F.l1_loss(scr_depth,target_depth,reduction='none')

        losses = {}
        losses['loss_depth'] = loss_depth.sum()/num_items

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, num_items, **kwargs):

        loss_map = {
            'juncs_labels': self.loss_junctions_labels,
            'juncs': self.loss_junctions,
            'depth':self.loss_depth
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_items, **kwargs)

    def forward(self, outputs, targets, origin_indices=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        losses = {}
        for i in range(len(outputs_without_aux['pred_logits'])):
            temp = {
                'pred_logits':outputs_without_aux['pred_logits'][i],
                'pred_juncs': outputs_without_aux['pred_juncs'][i],
                'pred_depth': outputs_without_aux['pred_depth'][i]
            }
            origin_indices = self.matcher(temp, targets)
            num_items = sum(len(t["labels"]) for t in targets)

            num_items = torch.as_tensor([num_items], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_items)
            num_items = torch.clamp(num_items / get_world_size(), min=1).item()
            # Compute all the requested losses
            for loss in self.losses:
                l = self.get_loss(loss, temp, targets, num_items, origin_indices=origin_indices)
                k = None
                v = None
                for kk,vv in l.items():
                    k = kk
                    v = vv
                if k not in losses.keys():
                    losses.update(l)
                else:
                    losses[k] += v

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        aux_name = 'aux_outputs'
        if aux_name in outputs:
            for i, aux_outputs in enumerate(outputs[aux_name]):
                for j in range(len(aux_outputs['pred_logits'])):
                    temp={
                        'pred_logits': aux_outputs['pred_logits'][j],
                        'pred_juncs': aux_outputs['pred_juncs'][j],
                        'pred_depth': aux_outputs['pred_depth'][j]
                    }

                    origin_indices = self.matcher(temp, targets)

                    for loss in self.losses:
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}

                        l = self.get_loss(loss, temp, targets, num_items, origin_indices=origin_indices,
                                               **kwargs)
                        l = {k + f'_{i}': v for k, v in l.items()}
                        k = None
                        v = None
                        for kk, vv in l.items():
                            k = kk
                            v = vv
                        if k not in losses.keys():
                            losses.update(l)
                        else:
                            losses[k] += v

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_criterion(cfg,weight_dict,matcher,device):

    losses = []
    losses.append('juncs_labels')
    losses.append('juncs')
    losses.append('depth')
    aux_layer = cfg.model.dec_layers
    if cfg.aux_loss:
        aux_weight_dict = {}
        for i in range(aux_layer - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(cfg.num_class, weight_dict, cfg.eos_coef, losses,eval(cfg.label_loss_params),
                             cfg.label_loss_func, matcher=matcher)
    criterion.to(device)

    return criterion