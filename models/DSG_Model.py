import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import mpl_toolkits.mplot3d as p3d
import matplotlib
import matplotlib.pyplot as plt
from models.GResnet.GCN import GCN
from models.position_encoding import build_position_encoding,build_position_encoding_3D
from models.transformer import build_transformer
from models.HRNet import build_hrnet
from encoder.hafm import HAFMencoder
from utils.net_utils import *
logger = logging.getLogger(__name__)
use_biase = False
use_align_corners = False

def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)
    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])
    return loss.mean()

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff,  jdepth, topk = 300, th=0.0):
    #import pdb;pdb.set_trace()
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    jdepth = jdepth.reshape(-1)
    joff = joff.reshape(2, -1)
    scores, index = torch.topk(jloc, k=topk)
    y = (index / width).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5
    z = torch.gather(jdepth,0,index)
    junctions = torch.stack((x, y)).t()

    return junctions[scores>th], scores[scores>th], z[scores>th]

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)
    if mask is not None:
        w = mask.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(mask/w)
    return loss.mean()

def weighted_l1_loss(input,target):
    mask = (target != 0).float()
    if mask.sum()!= 0:
        loss = F.l1_loss(input,target,reduce=None) * mask
        return loss.sum()/mask.sum()
    else:
        return 0

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

class DSG_Model(nn.Module):
    def __init__(self, cfg, position_embedding_mode='sine'):
        super(DSG_Model, self).__init__()
        self.stage = cfg.stage
        self.hidden_dim = 256
        self.backbone = build_hrnet(cfg)
        self.backbone_channels = self.backbone.out_channels
        self.hafm_encoder = HAFMencoder(cfg)
        self.n_dyn_junc = cfg.hawp.N_DYN_JUNC
        self.n_dyn_posl = cfg.hawp.N_DYN_POSL
        self.n_dyn_negl = cfg.hawp.N_DYN_NEGL
        self.n_dyn_othr = cfg.hawp.N_DYN_OTHR
        self.n_dyn_othr2 = cfg.hawp.N_DYN_OTHR2
        self.n_pts0 = cfg.hawp.N_PTS0
        self.n_pts1 = cfg.hawp.N_PTS1
        self.dim_loi = cfg.hawp.DIM_LOI
        self.dim_fc = cfg.hawp.DIM_FC
        self.n_out_junc = cfg.hawp.N_OUT_JUNC
        self.n_out_line = cfg.hawp.N_OUT_LINE
        self.use_residual = cfg.hawp.USE_RESIDUAL
        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None, None, :].cuda())
        self.num_sample_pts = cfg.model.num_sample_pts
        self.fc1 = nn.Conv2d(self.backbone_channels[0], 128, 1)
        self.pool1d = nn.MaxPool1d(self.n_pts0//self.n_pts1, self.n_pts0//self.n_pts1)
        self.fc2 = nn.Sequential(
            nn.Linear(128 * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, 1),
        )
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        if self.stage >= 2:
        # Transformer Branch
            self.aux_loss = cfg.aux_loss
            if self.aux_loss:
                self.return_inter = True
            else:
                self.return_inter = False
            self.use_context = cfg.model.use_context
            self.num_class = cfg.num_class
            self.num_queries = cfg.model.num_queries
            self.context_channels = self.backbone_channels[3]
            self.line_channels = self.backbone_channels[cfg.model.line_channel]
            self.lines_reduce = nn.Sequential(
                nn.Linear(self.hidden_dim * self.num_sample_pts, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim//2), )
            self.max_vlines = 40
            self.channel = 64
            if self.use_context:
                self.input_proj = nn.Conv2d(self.context_channels, self.hidden_dim, kernel_size=1)
            self.lines_proj = nn.Conv2d(self.line_channels, self.hidden_dim, kernel_size=1)
            self.position_embedding_2D = build_position_encoding(position_embedding_mode, hidden_dim=self.hidden_dim)
            self.line_position_embedding_layer = nn.Linear(4, self.hidden_dim//2)
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
            self.transformer = build_transformer(hidden_dim=self.hidden_dim, dropout=0.1, nheads=8, dim_feedforward=1024,
                                                 enc_layers=cfg.model.enc_layers, dec_layers=cfg.model.dec_layers, pre_norm=True, return_inter=self.return_inter,
                                                 use_context=self.use_context, aux_loss=self.aux_loss)
            self.class_embed = nn.Linear(self.hidden_dim, self.num_class + 1)
            self.juncs_embed = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
            self.depth_embed = MLP(self.hidden_dim,self.hidden_dim,1, 3)
        if self.stage >= 3:
        # Hidden line verification branch
            self.linep_channels = self.backbone_channels[0]
            self.lines_proj_p = nn.Conv2d(self.linep_channels, self.hidden_dim, kernel_size=1)
            self.lines_reduce_p = nn.Sequential(
                nn.MaxPool1d(8, 8),
                nn.Linear(self.line_channels * self.num_sample_pts // 8, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim))
            self.num_static_lines = 100
            self.num_dynamic_lines = 200

            self.verfication_embed = nn.Sequential(
                nn.Linear(self.hidden_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1),
            )
        if self.stage >= 4:
            # GCN Branch
            self.reprojection_matrix = torch.tensor([(1.5625, 0.0000,  0.0000,  0.0000),
                                                    (0.0000, 1.5625,  0.0000,  0.0000),
                                                    (0.0000, 0.0000, -1.0020, -0.2002),
                                                    (0.0000, 0.0000, -1.0000,  0.0000)],
                                                    dtype=torch.float, device='cuda').inverse()
            self.max_nodes = cfg.max_nodes
            self.gcn = GCN(state_dim=cfg.gcn.state_dim,feature_dim=self.backbone_channels[0]+3, out_dim=3, layer_num=cfg.gcn.layer_num,use_residual=cfg.gcn.use_residual)

    def loss_hafm(self,scores,targets,metas,features,annotations):
        loss_dict = {
            'loss_md': 0.0,
            'loss_dis': 0.0,
            'loss_res': 0.0,
            'loss_jloc_v': 0.0,
            'loss_joff_v': 0.0,
            'loss_jdepth_v': 0.0,
            'loss_jloc_h': 0.0,
            'loss_joff_h': 0.0,
            'loss_jdepth_h': 0.0,
            'loss_jloc_vh': 0.0,
            'loss_joff_vh': 0.0,
            'loss_jdepth_vh': 0.0,
            'loss_pos': 0.0,
            'loss_neg': 0.0
        }

        mask_visi = targets['mask_visi']
        loss_map = torch.mean(F.l1_loss(scores[:, :3].sigmoid(), targets['md_visi'], reduction='none'), dim=1, keepdim=True)
        loss_dict['loss_md'] += torch.mean(loss_map * mask_visi) / torch.mean(mask_visi)
        loss_map = F.l1_loss(scores[:, 3:4].sigmoid(), targets['dis_visi'], reduction='none')
        loss_dict['loss_dis'] += torch.mean(loss_map * mask_visi) / torch.mean(mask_visi)
        loss_residual_map = F.l1_loss(scores[:, 4:5].sigmoid(), loss_map, reduction='none')
        loss_dict['loss_res'] += torch.mean(loss_residual_map * mask_visi) / torch.mean(mask_visi)
        loss_dict['loss_jloc_v'] += cross_entropy_loss_for_junction(scores[:, 5:7], targets['jloc_v'])
        loss_dict['loss_joff_v'] += sigmoid_l1_loss(scores[:, 7:9], targets['joff_v'], -0.5,targets['jloc_v'])
        loss_dict['loss_jdepth_v'] += weighted_l1_loss(scores[:, 9].unsqueeze(1), targets['jdepth_v'])
        loss_dict['loss_jloc_h'] += cross_entropy_loss_for_junction(scores[:, 10:12], targets['jloc_h'])
        loss_dict['loss_joff_h'] += sigmoid_l1_loss(scores[:, 12:14], targets['joff_h'], -0.5, targets['jloc_h'])
        loss_dict['loss_jdepth_h'] += weighted_l1_loss(scores[:, 14].unsqueeze(1), targets['jdepth_h'])
        loss_dict['loss_jloc_vh'] += cross_entropy_loss_for_junction(scores[:, 15:17], targets['jloc_vh'])
        loss_dict['loss_joff_vh'] += sigmoid_l1_loss(scores[:, 17:19], targets['joff_vh'], -0.5,targets['jloc_vh'])
        loss_dict['loss_jdepth_vh'] += weighted_l1_loss(scores[:, 19].unsqueeze(1), targets['jdepth_vh'])
        loi_features = self.fc1(features)
        md_pred = scores[:, :3].sigmoid()
        dis_pred = scores[:, 3:4].sigmoid()
        res_pred = scores[:, 4:5].sigmoid()
        jloc_pred_v = scores[:, 5:7].softmax(1)[:, 1:]
        joff_pred_v = scores[:, 7:9].sigmoid() - 0.5
        jdepth_pred_v = scores[:,9].unsqueeze(1)
        jloc_pred_vh = scores[:, 15:17].softmax(1)[:, 1:]
        joff_pred_vh = scores[:, 17:19].sigmoid() - 0.5
        jdepth_pred_vh = scores[:,19].unsqueeze(1)
        batch_size = md_pred.size(0)
        lines_visi_pred = []
        idx_lines_for_junctions_visi = []
        junctions_cc_matched = []
        junctions_pred = []
        zs_pred = []
        for i, (md_pred_per_im, dis_pred_per_im, res_pred_per_im, meta) in enumerate(zip(md_pred, dis_pred, res_pred, metas)):
            lines_pred = []
            if self.use_residual:
                for scale in [-1.0, 0.0, 1.0]:
                    _ = proposal_lines(md_pred_per_im, dis_pred_per_im + scale * res_pred_per_im).view(-1, 4)
                    lines_pred.append(_)
            else:
                lines_pred.append(self.proposal_lines(md_pred_per_im, dis_pred_per_im).view(-1, 4))
            ann = annotations[i]
            lines_pred = torch.cat(lines_pred)
            junction_gt_vh = meta['junc_vh']
            junction_gt_v = meta['junc_v']
            junction_gt = torch.cat((junction_gt_vh, junction_gt_v), dim=0)
            juncs_cc_gt = ann['junctions_cc']
            junctions_label = ann['junction_label']
            juncs_cc_vh_gt = juncs_cc_gt[junctions_label==3]
            juncs_cc_v_gt = juncs_cc_gt[junctions_label==1]
            juncs_cc_gt = torch.cat([juncs_cc_vh_gt,juncs_cc_v_gt],dim=0)
            N_vh = junction_gt_vh.size(0)
            N_v = junction_gt_v.size(0)
            juncs_pred_vh, _, z_pred_vh = get_junctions(non_maximum_suppression(jloc_pred_vh[i]), joff_pred_vh[i], jdepth_pred_vh[i],
                                             topk=min(N_vh * 2 + 2, 20))
            juncs_pred_v, _, z_pred_v = get_junctions(non_maximum_suppression(jloc_pred_v[i]), joff_pred_v[i], jdepth_pred_v[i],
                                           topk=min(N_v * 2 + 2, 20))
            juncs_pred = torch.cat((juncs_pred_vh, juncs_pred_v), dim=0)
            z_pred = torch.cat([z_pred_vh,z_pred_v],dim=0)
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1).min(
                0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(
                0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)
            iskeep = idx_junc_to_end_min < idx_junc_to_end_max
            idx_lines_for_junctions = torch.cat((idx_junc_to_end_min[iskeep, None], idx_junc_to_end_max[iskeep, None]),
                                                dim=1).unique(dim=0)
            try:
                idx_lines_for_junctions_mirror = torch.cat(
                    (idx_lines_for_junctions[:, 1, None], idx_lines_for_junctions[:, 0, None]), dim=1)
                idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
            except:
                import pdb;pdb.set_trace()
            lines_adjusted = torch.cat(
                (juncs_pred[idx_lines_for_junctions[:, 0]], juncs_pred[idx_lines_for_junctions[:, 1]]), dim=1)

            cost_, match_ = torch.sum((juncs_pred - junction_gt[:, None]) ** 2, dim=-1).min(0)
            juncs_cc_gt = torch.cat([juncs_cc_gt,torch.zeros((1,3),dtype=torch.float,device='cuda')],dim=0)
            match_[cost_ > 1.5 * 1.5] = N_vh + N_v
            juncs_cc_match = juncs_cc_gt[match_]
            junctions_cc_matched.append(juncs_cc_match)
            Lpos = meta['Lpos_visi']
            labels = Lpos[match_[idx_lines_for_junctions[:, 0]], match_[idx_lines_for_junctions[:, 1]]]
            iskeep = torch.zeros_like(labels, dtype=torch.bool)
            cdx = labels.nonzero().flatten()

            if len(cdx) > self.n_dyn_posl:
                perm = torch.randperm(len(cdx), device='cuda')[:self.n_dyn_posl]
                cdx = cdx[perm]

            iskeep[cdx] = 1

            if self.n_dyn_othr2 > 0:
                cdx = (labels == 0).nonzero().flatten()
                if len(cdx) > self.n_dyn_othr2:
                    perm = torch.randperm(len(cdx), device='cuda')[:self.n_dyn_othr2]
                    cdx = cdx[perm]
                iskeep[cdx] = 1

            lines_selected = lines_adjusted[iskeep]
            num_lines_selected = lines_selected.shape[0]
            labels_selected = labels[iskeep]
            idx_lines_for_junctions = idx_lines_for_junctions[iskeep]
            lines_for_train = torch.cat((lines_selected, meta['lpre_visi']))
            labels_for_train = torch.cat((labels_selected.float(), meta['lpre_label_visi']))
            logits = self.pooling(loi_features[i], lines_for_train)
            loss_ = self.loss(logits, labels_for_train)
            idx_lines_for_junctions = idx_lines_for_junctions[logits[:num_lines_selected].sigmoid() > 0.5]
            idx_lines_for_junctions = idx_lines_for_junctions[idx_lines_for_junctions[:,0] < idx_lines_for_junctions[:,1]]
            idx_lines_for_junctions_visi.append(idx_lines_for_junctions)
            lines_visi_pred.append(torch.cat((juncs_pred[idx_lines_for_junctions[:, 0]], juncs_pred[idx_lines_for_junctions[:, 1]]), dim=1))
            junctions_pred.append(juncs_pred)
            zs_pred.append(z_pred)
            loss_positive = loss_[labels_for_train == 1].mean()
            loss_negative = loss_[labels_for_train == 0].mean()
            loss_dict['loss_pos'] += loss_positive / batch_size
            loss_dict['loss_neg'] += loss_negative / batch_size

        return loss_dict, lines_visi_pred, idx_lines_for_junctions_visi,junctions_cc_matched, junctions_pred,z_pred

    def proposal_lines_hidden(self,juncs_vh_pred, juncs_h_pred,ann, meta):

        juncs_gt = ann['junctions_2D']/2
        juncs_cc_gt = ann['junctions_cc']
        juncs_label = ann['junction_label']
        juncs_cc_gt = juncs_cc_gt[juncs_label==2]
        juncs_vh_gt = juncs_gt[juncs_label==3]
        juncs_h_gt = juncs_gt[juncs_label==2]
        num_vh_pred = juncs_vh_pred.shape[0]
        num_h_pred = juncs_h_pred.shape[0]
        num_vh_gt = juncs_vh_gt.shape[0]
        num_h_gt = juncs_h_gt.shape[0]
        hi = (torch.arange(num_h_pred) + num_vh_pred).unsqueeze(1).expand(-1,(num_vh_pred + num_h_pred))
        vhi = (torch.arange(num_vh_pred + num_h_pred)).unsqueeze(0).expand(num_h_pred, -1)
        ls = torch.stack([vhi, hi], dim=-1).flatten(0, 1)
        cost_h, match_h = torch.sum((juncs_h_pred - juncs_h_gt[:,None])**2,dim=-1).min(0)
        juncs_cc_gt = torch.cat([juncs_cc_gt,torch.zeros((1,3),dtype=torch.float,device='cuda')],dim=0)
        juncs_cc_matched = juncs_cc_gt[match_h]
        match_h[cost_h > 2 * 2] = num_h_gt
        cost_vh, match_vh = torch.sum((juncs_vh_pred - juncs_vh_gt[:,None])**2,dim=-1).min(0)
        match_vh[cost_vh>1.5*1.5] =num_vh_gt + num_h_gt
        match_hi = match_h.unsqueeze(1).expand(-1,match_h.shape[0] + match_vh.shape[0]) + num_vh_gt
        match_vhi = torch.cat([match_vh,match_h+num_vh_gt],dim=0).unsqueeze(0).expand(match_h.shape[0],-1)
        ls_match = torch.stack([match_hi,match_vhi],dim=-1).flatten(0,1)
        label = meta['Lpos_hidden'][ls_match[:,0],ls_match[:,1]]
        idx = ls[:,0] < ls[:,1]
        ls = ls[idx]
        label = label[idx].float()
        juncs = torch.cat([juncs_vh_pred,juncs_h_pred],dim=0)
        lines_proposal = juncs[ls].reshape((-1,4))
        lpre = meta['lpre_hidden']
        lpre_label = meta['lpre_label_hidden']
        lines_proposal = torch.cat([lines_proposal,lpre],dim=0)
        labels = torch.cat([label,lpre_label],dim=0)

        return lines_proposal,labels,juncs_cc_matched,ls.cuda()

    def verfiy_line_visi(self, loi_features, md_pred, dis_pred, res_pred, juncs_pred):

        if self.use_residual:
            lines_pred = proposal_lines_new(md_pred[0], dis_pred[0], res_pred[0]).view(-1, 4)
        else:
            lines_pred = proposal_lines_new(md_pred[0], dis_pred[0], None).view(-1, 4)

        dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1).min(0)
        dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

        idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)

        iskeep = (
                    idx_junc_to_end_min < idx_junc_to_end_max)  # * (dis_junc_to_end1< 10*10)*(dis_junc_to_end2<10*10)  # *(dis_junc_to_end2<100)

        idx_lines_for_junctions = torch.unique(
            torch.cat((idx_junc_to_end_min[iskeep, None], idx_junc_to_end_max[iskeep, None]), dim=1),
            dim=0)
        lines_adjusted = torch.cat(
            (juncs_pred[idx_lines_for_junctions[:, 0]], juncs_pred[idx_lines_for_junctions[:, 1]]), dim=1)

        scores = self.pooling(loi_features[0], lines_adjusted).sigmoid()

        lines_final = lines_adjusted[scores > 0.05]
        score_final = scores[scores > 0.05]
        idx_lines_for_junctions = idx_lines_for_junctions[scores > 0.05]

        return lines_final, score_final, idx_lines_for_junctions

    def pooling(self, features_per_image, lines_per_im):
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) -0.5
        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)+ features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)+ features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)+ features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(128,-1,32)
        ).permute(1,0,2)
        # if self.pool1d is not None:
        xp = self.pool1d(xp)
        features_per_line = xp.view(-1, self.n_pts1*self.dim_loi)
        logits = self.fc2(features_per_line).flatten()

        return logits

    def forward(self,x, annotations=None):
        if self.training:
            return self.forward_train(x, annotations)
        else:
            return self.forward_test(x, annotations)

    def forward_test(self, x, annotations=None):
        with torch.no_grad():
            results = {}
            bs, _, ho, wo = x.shape
            y_list, scores = self.backbone(x)  # c: 32, 64, 128, 256
            c1, c2, c3, c4 = y_list
            md_pred = scores[:, :3].sigmoid()
            dis_pred = scores[:, 3:4].sigmoid()
            res_pred = scores[:, 4:5].sigmoid()
            jloc_pred_v = scores[:, 5:7].softmax(1)[:, 1:]
            joff_pred_v = scores[:, 7:9].sigmoid() - 0.5
            jdepth_pred_v = scores[:, 9].unsqueeze(1)
            jloc_pred_h = scores[:,10:12].softmax(1)[:, 1:]
            joff_pred_h = scores[:,12:14].sigmoid() - 0.5
            jdepth_pred_h = scores[:, 14].unsqueeze(1)
            jloc_pred_vh = scores[:, 15:17].softmax(1)[:, 1:]
            joff_pred_vh = scores[:, 17:19].sigmoid() - 0.5
            jdepth_pred_vh = scores[:, 19].unsqueeze(1)
            juncs_pred_vh, juncs_score_vh, z_pred_vh = get_junctions(non_maximum_suppression(jloc_pred_vh[0]), joff_pred_vh[0],
                                                        jdepth_pred_vh[0],topk=30, th=0.1)
            juncs_pred_v, juncs_score_v, z_pred_v = get_junctions(non_maximum_suppression(jloc_pred_v[0]), joff_pred_v[0],
                                                      jdepth_pred_v[0],topk= 30, th=0.1)
            juncs_pred_h, juncs_score_h, z_pred_h = get_junctions(non_maximum_suppression(jloc_pred_h[0]), joff_pred_h[0],
                                                                  jdepth_pred_h[0], topk=30, th=0.1)
            results['junctions_hidden_conv'] = juncs_pred_h.detach().cpu().numpy().tolist()
            results['junctions_hidden_conv_score'] = juncs_score_h.detach().cpu().numpy().tolist()
            juncs_pred = torch.cat((juncs_pred_vh, juncs_pred_v), dim=0)
            loi_feature = self.fc1(c1)
            lines_visi, scores_visi, idx_lines_for_junctions_visi = self.verfiy_line_visi(loi_feature, md_pred, dis_pred, res_pred, juncs_pred)
            results['junctions_visible_pred'] = juncs_pred_v.detach().cpu().tolist()
            results['junctions_visible_score'] = juncs_score_v.detach().cpu().tolist()
            results['junctions_vh_pred'] = juncs_pred_vh.detach().cpu().tolist()
            results['junctions_vh_score'] = juncs_score_vh.detach().cpu().tolist()
            results['lines_visible_pred'] = lines_visi.detach().cpu().tolist()
            results['lines_visible_score'] = scores_visi.detach().cpu().tolist()
            results['idx_lines_for_junctions_visi'] = idx_lines_for_junctions_visi.detach().cpu().tolist()
            if self.stage == 1:
                return results
            lines_visi = lines_visi[scores_visi>0.5]
            idx_lines_for_junctions_visi = idx_lines_for_junctions_visi[scores_visi>0.5]
            lines = torch.zeros((1, self.max_vlines, 4), dtype=torch.float, device='cuda')
            lines_visi = lines_visi[:self.max_vlines]
            num_lines = lines_visi.shape[0]
            lines[0,:num_lines] = lines_visi
            mask = torch.zeros((1,self.max_vlines),dtype=torch.bool, device=lines.device)
            mask[:,num_lines:] = True
            mask_temp = mask.unsqueeze(-1).float()
            src_line = c4
            line_proj = self.lines_proj(src_line)  # b, hidden_dim, h', w'
            lines_feat = get_lines_features(line_proj, lines, [128, 128],
                                            self.num_sample_pts)  # B, hidden_dim, max_line_num
            lines_feat = lines_feat.permute(0, 2, 1)  # B, max_line_num, hidden_dim
            lines_feat_pos = self.line_position_embedding_layer(lines)
            lines_feat = self.lines_reduce(lines_feat)
            lines_feat = torch.cat((lines_feat, lines_feat_pos), dim=-1)
            lines_pos = torch.zeros_like(lines_feat, dtype=torch.float, device='cuda')
            lines_feat = lines_feat * (1 - mask_temp)  # B, max_line_num, hidden_dim
            if self.use_context:
                src = c4
                src = self.input_proj(src)  # b, hidden_dim, h, w
                pos = self.position_embedding_2D(src)
                hs_all, _, memory, attn_weights = self.transformer(lines_feat, self.query_embed.weight, lines_pos, tgt=None,
                                                                   src=src, mask_lines=mask,
                                                                   pos_embed=pos)  # memory: b, c, h, w
            else:
                hs_all, _, memory, attn_weights = self.transformer(lines_feat, self.query_embed.weight, lines_pos, tgt=None,
                                                                   src=None, mask_lines=mask,
                                                                   pos_embed=None)  # memory: b, c, h, w
            # ------------------------------------------------------- line instance decoder
            hs = hs_all.contiguous()  # num_decode_layer, 3, bs, num_querry, hs
            outputs_class = self.class_embed(hs)
            outputs_coord = self.juncs_embed(hs).sigmoid()
            outputs_depth = self.depth_embed(hs)
            pred_logits = outputs_class[-1, -1, :]
            pred_logits = pred_logits.softmax(-1)
            pred_coord = outputs_coord[-1, -1, :]
            pred_depth = outputs_depth[-1, -1, :]
            p_l = pred_logits[0]
            p_c = pred_coord[0]
            p_d = pred_depth[0]
            idx = p_l[:,0] > 0.1
            juncs_pred_h = p_c[idx] * 128
            juncs_score_h = p_l[idx,0]
            z_pred_h = p_d[idx, 0]
            results['junctions_hidden_pred'] = juncs_pred_h.detach().cpu().tolist()
            results['junctions_score_hidden'] = juncs_score_h.detach().cpu().tolist()
            results['junctions_depth_hidden'] = z_pred_h.detach().cpu().tolist()
            if self.stage == 2:
                return results
            juncs_pred = torch.cat([juncs_pred_vh,juncs_pred_h],dim=0)
            num_juncs_vh = juncs_pred_vh.shape[0]
            num_juncs_h = juncs_pred_h.shape[0]
            hi = (torch.arange(num_juncs_h) + num_juncs_vh).unsqueeze(1).expand((-1, num_juncs_h + num_juncs_vh))
            vi = (torch.arange(num_juncs_vh + num_juncs_h)).unsqueeze(0).expand((num_juncs_h, -1))
            ls = torch.stack([vi, hi], dim=-1).flatten(0, 1)
            idx = ls[:, 0] < ls[:, 1]
            ls = ls[idx]
            idx_lines_for_junctions_hidden = ls
            lines_proposal_hidden = juncs_pred[ls].reshape((-1, 4))
            src_linesp = c1[0]
            src_linesp = self.lines_proj_p(src_linesp.unsqueeze(0))
            lines_featurep = get_lines_features(src_linesp, lines_proposal_hidden.unsqueeze(0), [128, 128],
                                                self.num_sample_pts)
            lines_featurep = lines_featurep.permute(0, 2, 1).contiguous()
            lines_featurep = self.lines_reduce_p(lines_featurep)
            logits = self.verfication_embed(lines_featurep).flatten().sigmoid()
            idx_lines_for_junctions_hidden[idx_lines_for_junctions_hidden >= juncs_pred_vh.shape[0]] = \
                idx_lines_for_junctions_hidden[idx_lines_for_junctions_hidden >= juncs_pred_vh.shape[0]] + \
                juncs_pred_v.shape[0]
            lines_hidden_pred = lines_proposal_hidden[logits > 0.05]
            idx_lines_for_junctions_hidden = idx_lines_for_junctions_hidden[logits > 0.05].cuda()
            logits = logits[logits > 0.05]
            results['lines_hidden_pred'] = lines_hidden_pred.detach().cpu().tolist()
            results['lines_hidden_score'] = logits.detach().cpu().tolist()
            results['idx_lines_for_junctions_hidden'] = idx_lines_for_junctions_hidden.detach().cpu().tolist()
            idx_lines_for_junctions_hidden = idx_lines_for_junctions_hidden[logits > 0.5]
            juncs_pred = torch.cat([juncs_pred_vh, juncs_pred_v, juncs_pred_h], dim=0)
            juncs_label_pred = torch.cat([torch.ones(juncs_pred_vh.shape[0])*3,torch.ones(juncs_pred_v.shape[0])*1, torch.ones(juncs_pred_h.shape[0])*2])
            depth_pred = torch.cat([z_pred_vh,z_pred_v,z_pred_h])
            junctions_score = torch.cat([juncs_score_vh,juncs_score_v,juncs_score_h])
            results['junctions_pred'] = juncs_pred.detach().cpu().tolist()
            results['junctions_label_pred'] = juncs_label_pred.tolist()
            results['junctions_score'] = junctions_score.detach().cpu().tolist()
            results['depth_pred'] = depth_pred.detach().cpu().tolist()
            if self.stage == 3:
                return results
            juncs_pad = torch.zeros((self.max_nodes, 2), dtype=torch.float, device='cuda')
            z_pad = torch.zeros((self.max_nodes, 1), dtype=torch.float, device='cuda')
            num_junc = juncs_pred.shape[0]
            juncs_pad[:num_junc] = juncs_pred
            z_pad[:num_junc] = depth_pred.unsqueeze(1)
            idx_lines_for_junctions = torch.cat([idx_lines_for_junctions_visi, idx_lines_for_junctions_hidden], dim=0)
            adj_matrix = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float, device='cuda')
            adj_matrix[idx_lines_for_junctions[:, 0], idx_lines_for_junctions[:, 1]] = 1
            adj_matrix[idx_lines_for_junctions[:, 1], idx_lines_for_junctions[:, 0]] = 1
            adj_matrix = adj_matrix + torch.eye(self.max_nodes, dtype=torch.float, device='cuda')
            normalized_adj_matrix = adj_matrix / (torch.sum(adj_matrix, dim=1, keepdim=True) \
                                                  + 1e-6).repeat(1, adj_matrix.shape[-1])

            scr_juncs = c1.reshape(c1.shape[0], c1.shape[1], -1).permute(0, 2, 1).contiguous()
            juncs_features = interpolated_sum(scr_juncs.unsqueeze(0), juncs_pad.unsqueeze(0) / 128, [c1.shape[-2:]])
            junctions_cc_pred = reprojection(self.reprojection_matrix, juncs_pad, z_pad)
            results['junctions_cc_pred'] = junctions_cc_pred[:num_junc].detach().cpu().tolist()
            juncs_features = torch.cat([juncs_features, junctions_cc_pred.unsqueeze(0)], dim=-1)
            residual_pred = self.gcn(juncs_features, normalized_adj_matrix.unsqueeze(0))
            results['junctions_cc_refined'] = (junctions_cc_pred + residual_pred[0])[:num_junc].detach().cpu().tolist()

        return results

    def forward_train(self, x,annotations=None):
        #stage 1
        bs, _, ho, wo = x.shape
        targets,meta = self.hafm_encoder(annotations)
        y_list,scores = self.backbone(x)  # c: 32, 64, 128, 256
        c1, c2, c3, c4 = y_list
        hafm_loss_dict, lines_visi_pred, idx_lines_for_junctions_visi,junctions_cc_matched_visi,junctions_pred, zs_pred\
            = self.loss_hafm(scores, targets, meta, y_list[0],annotations)
        if self.stage == 1:
            return  hafm_loss_dict, None, None, None
        #stage 2
        lines = torch.zeros((bs,self.max_vlines,4),dtype=torch.float,device='cuda')
        max_line_num = self.max_vlines
        masks = []
        for i,ls in enumerate(lines_visi_pred):
            ls = ls[:self.max_vlines]
            valid_line_num = int(ls.shape[0])
            lines[i,:valid_line_num] = ls
            mask_i = torch.zeros((1, max_line_num), dtype=torch.bool, device=lines.device)  # 1, max_line_num
            mask_i[:, valid_line_num:] = True
            masks.append(mask_i)
        mask_lines = torch.cat(masks, dim=0).contiguous()  # b, max_line_num
        mask_temp = mask_lines.unsqueeze(-1).float()
        src_line = c4
        line_proj = self.lines_proj(src_line)  # b, hidden_dim, h', w'
        lines_feat = get_lines_features(line_proj,lines,[128, 128], self.num_sample_pts)  # B, hidden_dim, max_line_num
        lines_feat = lines_feat.permute(0, 2, 1)  # B, max_line_num, hidden_dim
        lines_feat_pos = self.line_position_embedding_layer(lines)
        lines_feat = self.lines_reduce(lines_feat)
        lines_feat = torch.cat((lines_feat, lines_feat_pos), dim=-1)
        lines_pos = torch.zeros_like(lines_feat, dtype=torch.float, device='cuda')
        lines_feat = lines_feat * (1 - mask_temp)  # B, max_line_num, hidden_dim
        if self.use_context:
            src = c4
            src = self.input_proj(src)  # b, hidden_dim, h, w
            pos = self.position_embedding_2D(src)
            hs_all, _, memory, attn_weights = self.transformer(lines_feat, self.query_embed.weight, lines_pos, tgt=None,
                                                       src=src, mask_lines=mask_lines,
                                                       pos_embed=pos)  # memory: b, c, h, w
        else:
            hs_all, _, memory, attn_weights = self.transformer(lines_feat, self.query_embed.weight, lines_pos, tgt=None,
                                                       src=None, mask_lines=mask_lines,
                                                       pos_embed=None)  # memory: b, c, h, w
        # ------------------------------------------------------- line instance decoder
        hs = hs_all.contiguous()  # num_decode_layer, 3, bs, num_querry, hs
        outputs_class = self.class_embed(hs)
        outputs_coord = self.juncs_embed(hs).sigmoid()
        outputs_depth = self.depth_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_juncs': outputs_coord[-1], 'pred_depth':outputs_depth[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_depth)
        if self.stage == 2:
            return hafm_loss_dict, out, None, None
        #stage 3
        pred_logits = outputs_class[-1, -1, :]
        pred_logits = pred_logits.softmax(-1)
        pred_coord = outputs_coord[-1, -1, :]
        pred_depth = outputs_depth[-1, -1, :]
        jloc_v = scores[:, 5:7].softmax(1)[:, 1:]
        joff_v = scores[:, 7:9].sigmoid() - 0.5
        jdepth_v = scores[:, 9].unsqueeze(1)
        jloc_vh = scores[:, 15:17].softmax(1)[:, 1:]
        joff_vh = scores[:, 17:19].sigmoid() - 0.5
        jdepth_vh = scores[:, 19].unsqueeze(1)
        loss_hidden_lines = {
            'loss_pos':0.0,
            'loss_neg':0.0
        }
        adj_matrixes = []
        junctions = []
        zs = []
        num_junctions = []
        lines_idx = []
        cc_gt = []
        for i in range(bs):
            ann = annotations[i]
            label = ann['junction_label']
            N_v = (label == 1).sum()
            N_h = (label == 2).sum()
            N_vh = (label == 3).sum()
            juncs_vh_pred, _, z_vh_pred = get_junctions(non_maximum_suppression(jloc_vh[i]), joff_vh[i],
                                                        jdepth_vh[i],
                                                        topk=min(N_vh * 2 + 2, 20))
            juncs_v_pred, _, z_v_pred = get_junctions(non_maximum_suppression(jloc_v[i]), joff_v[i],
                                                      jdepth_v[i],
                                                      topk=min(N_v * 2 + 2, 20))
            p_l = pred_logits[i]
            p_c = pred_coord[i]
            p_d = pred_depth[i]
            _, idx = torch.topk(p_l[:,0],min(N_h*2+2,20))
            juncs_h_pred = p_c[idx] * 128
            z_h_pred = p_d[idx,0]
            lines_proposal_hidden, lines_label_hidden, juncs_cc_matched, idx_lines_for_junctions_hidden,  = self.proposal_lines_hidden(juncs_vh_pred,juncs_h_pred,ann,meta[i])
            src_linesp  = c1[i]
            src_linesp = self.lines_proj_p(src_linesp.unsqueeze(0))
            lines_featurep = get_lines_features(src_linesp,lines_proposal_hidden.unsqueeze(0), [128,128],self.num_sample_pts)
            lines_featurep = lines_featurep.permute(0, 2, 1).contiguous()
            lines_featurep = self.lines_reduce_p(lines_featurep)
            logits = self.verfication_embed(lines_featurep).flatten()
            loss = self.loss(logits,lines_label_hidden)
            loss_hidden_lines['loss_pos'] += loss[lines_label_hidden==1].mean()/bs
            loss_hidden_lines['loss_neg'] += loss[lines_label_hidden==0].mean()/bs
            if self.stage == 4:
                #import pdb;pdb.set_trace()
                logits = logits[:idx_lines_for_junctions_hidden.shape[0]].sigmoid()
                idx_lines_for_junctions_hidden = idx_lines_for_junctions_hidden[logits>0.5]
                idx_lines_for_junctions_hidden[idx_lines_for_junctions_hidden >= juncs_vh_pred.shape[0]] = \
                            idx_lines_for_junctions_hidden[idx_lines_for_junctions_hidden >= juncs_vh_pred.shape[0]] + juncs_v_pred.shape[0]
                juncs_pred = torch.cat([juncs_vh_pred,juncs_v_pred,juncs_h_pred],dim=0)
                juncs_cc_matched = torch.cat([junctions_cc_matched_visi[i],juncs_cc_matched],dim=0)
                z_pred = torch.cat([z_vh_pred,z_v_pred,z_h_pred]).unsqueeze(1)
                juncs_pad = torch.zeros((self.max_nodes,2),dtype=torch.float,device='cuda')
                z_pad = torch.zeros((self.max_nodes,1),dtype=torch.float,device='cuda')
                juncs_cc_matched_pad = torch.zeros((self.max_nodes,3),dtype=torch.float,device='cuda')
                num_junc = juncs_pred.shape[0]
                juncs_pad[:num_junc] = juncs_pred
                z_pad[:num_junc] = z_pred
                #import pdb;pdb.set_trace()
                juncs_cc_matched_pad[:num_junc] = juncs_cc_matched
                idx_lines_for_junctions = torch.cat([idx_lines_for_junctions_visi[i], idx_lines_for_junctions_hidden], dim=0)
                adj_matrix = torch.zeros((self.max_nodes,self.max_nodes),dtype=torch.float,device='cuda')
                adj_matrix[idx_lines_for_junctions[:,0],idx_lines_for_junctions[:,1]] = 1
                adj_matrix[idx_lines_for_junctions[:, 1], idx_lines_for_junctions[:, 0]] = 1
                adj_matrix = adj_matrix + torch.eye(self.max_nodes, dtype=torch.float, device='cuda')
                normalized_adj_matrix = adj_matrix / (torch.sum(adj_matrix, dim=1, keepdim=True)  \
                                                      + 1e-6).repeat(1,adj_matrix.shape[-1])
                adj_matrixes.append(normalized_adj_matrix)
                junctions.append(juncs_pad)
                zs.append(z_pad)
                lines_idx.append(idx_lines_for_junctions)
                cc_gt.append(juncs_cc_matched_pad)
                num_junctions.append(num_junc)

        if self.stage == 3:
            return hafm_loss_dict, out, loss_hidden_lines, None
        if self.stage == 4:
            junctions = torch.stack(junctions,dim=0)
            zs = torch.stack(zs,dim=0)
            adj_matrixes = torch.stack(adj_matrixes,dim=0)
            cc_gt = torch.stack(cc_gt,dim=0)
            scr_juncs = c1.reshape(c1.shape[0], c1.shape[1],-1).permute(0,2,1).contiguous()
            juncs_features = interpolated_sum(scr_juncs.unsqueeze(0),junctions/128,[c1.shape[-2:]])
            junctions_cc_pred = torch.stack([reprojection(self.reprojection_matrix,juncs,depth) for juncs,depth in zip(junctions,zs)])
            juncs_features = torch.cat([juncs_features,junctions_cc_pred],dim=-1)
            residual_gt = cc_gt - junctions_cc_pred[:,:,:3]
            residual_pred = self.gcn(juncs_features,adj_matrixes)
            return hafm_loss_dict, out, loss_hidden_lines, [residual_gt,residual_pred, num_junctions]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_depth):
        return [{'pred_logits': a, 'pred_juncs': b, 'pred_depth': c} for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_depth[:-1])]


