import os
import time
import random
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as tf
from utils.utils import Set_Config, Set_Logger, Set_Ckpt_Code_Debug_Dir
from models.DSG_Model import DSG_Model
from utils.misc import AverageMeter, get_optimizer
from models.matcher import build_matcher
from models.Criterion import build_criterion
from models.ABC_3DLInesDateset import build_dataset,collate_fn
import matplotlib
matplotlib.use('Agg')
import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str,
                    help='train / eval')
parser.add_argument('--backbone', default='hrnet', type=str,
                    help='only support hrnet now')
parser.add_argument('--cfg_path', default='configs/config_LineTR_train_s1.yaml', type=str,
                    help='full path of the config file')
args = parser.parse_args()
NUM_GPUS = torch.cuda.device_count()

torch.backends.cudnn.benchmark = True


class LossReducer(object):
    def __init__(self, cfg):
        # self.loss_keys = cfg.MODEL.LOSS_WEIGHTS.keys()
        self.loss_weights = dict(cfg.hawp.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k] * loss_dict[k]
                          for k in self.loss_weights.keys()])

        return total_loss


def loss_offset(pred,gt,num_juncs):
    mask = torch.zeros((pred.shape[0],pred.shape[1],1),dtype=torch.float,device='cuda')
    for i,n in enumerate(num_juncs):
        mask[i,:n] = 1
    loss = (F.l1_loss(pred,gt,reduction='none') * mask).sum()/mask.sum()

    return loss

def load_dataset(cfg, args):

    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.05822601, 0.05822601, 0.05822601], [0.16618816, 0.16618816, 0.16618816])
    ])
    assert NUM_GPUS > 0

    dataset = build_dataset('train',transforms)
    loaders = data.DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    data_sampler = None

    return loaders, data_sampler

def train(cfg, logger):
    logger.info('*' * 40)
    localtime = time.asctime(time.localtime(time.time()))
    logger.info(localtime)
    logger.info('start training......')
    logger.info('*' * 40)

    model_name = (cfg.save_path).split('/')[-1]

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set ckpt/code/debug dir to save
    checkpoint_dir = Set_Ckpt_Code_Debug_Dir(cfg, args, logger)

    # build network
    network = DSG_Model(cfg)
    network = network.to(device)

    # load pretrained weights if existed
    if not (cfg.resume_dir == 'None'):
        print('loading checkpoint from ' + cfg.resume_dir)
        model_dict = network.state_dict()
        pretrained_dict = torch.load(cfg.resume_dir)
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)

    optimizer = get_optimizer(network.parameters(),cfg.solver)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.solver.lr_step, gamma=cfg.solver.gamma)
    data_loader, data_sampler = load_dataset(cfg, args)
    network.train(not cfg.model.fix_bn)
    loss_reducer = LossReducer(cfg)
    writer = SummaryWriter('tensorboard_s{}'.format(cfg.stage))
    start_epoch = 0
    if cfg.stage >= 2:
        matcher = build_matcher(cost_class=1., cost_junc=5.)
        weight_dict = {'loss_ce': cfg.loss_ce, "loss_junc": cfg.junc_loss_coef, "loss_depth":cfg.depth_loss_coef}
        criterion = build_criterion(cfg,weight_dict,matcher,device)
        logger.info(f"used losses = {weight_dict}")
    for epoch in range(start_epoch, cfg.num_epochs):
        # --------------------------------------  time log
        batch_time = AverageMeter()

        # --------------------------------------  loss log
        losses = AverageMeter()
        metric_tracker = {'Offset_prediction': ('loss_o',AverageMeter()),
                         'Classify_instance': ('loss_ce', AverageMeter()),
                          'Regression_Juncs': ('loss_junc', AverageMeter()),
                          'Regression_Depth': ('loss_depth', AverageMeter()),
                          'LOSS_Hidden_pos': ('loss_pos', AverageMeter()),
                          'LOSS_Hidden_neg': ('loss_neg',AverageMeter())
                          }
        metric_tracker_hafm = {'LOSS_HAFM':('loss_final_hafm',AverageMeter()),
            'LOSS_MD_VISI':('loss_md',AverageMeter()),
            'LOSS_DIS_VISI': ('loss_dis',AverageMeter()),
            'LOSS_RES_VISI': ('loss_res', AverageMeter()),
            'LOSS_JLOC_V': ('loss_jloc_v', AverageMeter()),
            'LOSS_JOFF_V': ('loss_joff_v', AverageMeter()),
            'LOSS_JDEPTH_V': ('loss_jdepth_v', AverageMeter()),
            'LOSS_JLOC_H': ('loss_jloc_h', AverageMeter()),
            'LOSS_JOFF_H': ('loss_joff_h', AverageMeter()),
            'LOSS_JDEPTH_H':('loss_jdepth_h', AverageMeter()),
            'LOSS_JLOC_VH': ('loss_jloc_vh', AverageMeter()),
            'LOSS_JOFF_VH': ('loss_joff_vh', AverageMeter()),
            'LOSS_JDEPTH_VH': ('loss_jdepth_vh',AverageMeter()),
            'LOSS_Visible_pos':('loss_pos', AverageMeter()),
            'LOSS_Visible_neg':('loss_neg',AverageMeter())
        }
        tic = time.time()
        for iter, (sample,targets) in enumerate(data_loader):
            image = sample['image'].to(device).float()  # b, 3, h, w
            for i, t in enumerate(targets):
                for k, v in t.items():
                    if torch.is_tensor(v):
                        targets[i][k] = v.to(device)
            hafm_loss_dict, outputs, loss_hidden_lines, gcn_dict = network(image,targets)
            loss_final_hafm = loss_reducer(hafm_loss_dict)
            hafm_loss_dict['loss_final_hafm'] = loss_final_hafm
            loss_final = cfg.weight_hafm * loss_final_hafm
            if cfg.stage >= 2:
                loss_dict = criterion(outputs,targets)
                weight_dict = criterion.weight_dict
                loss_final_TR = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss_final = loss_final + cfg.weight_invTR * loss_final_TR
            if cfg.stage >= 3:
                loss_final = loss_final + cfg.weight_hidden_line_veri * (
                            loss_hidden_lines['loss_pos'] + loss_hidden_lines['loss_neg'])
                loss_dict.update(loss_hidden_lines)
            if cfg.stage == 4:
                residual_gt, residual_pred, num_junctions = gcn_dict
                loss_o = loss_offset(residual_pred, residual_gt, num_junctions)
                loss_final = loss_final + cfg.weight_GCN * loss_o
                loss_dict.update({'loss_o': loss_o})
            # --------------------------------------  Backward
            optimizer.zero_grad()
            loss_final.backward()
            # if cfg.clip_max_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.clip_max_norm)
            optimizer.step()
            # --------------------------------------  update losses and metrics
            losses.update(loss_final.item())
            if cfg.stage >= 2:
                for name_log in metric_tracker.keys():
                    name_loss = metric_tracker[name_log][0]
                    if name_loss in loss_dict.keys():
                        loss_cur = float(loss_dict[name_loss])
                        metric_tracker[name_log][1].update(loss_cur)

            for name_log in metric_tracker_hafm.keys():
                name_loss = metric_tracker_hafm[name_log][0]
                if name_loss in hafm_loss_dict.keys():
                    loss_cur = float(hafm_loss_dict[name_loss])
                    metric_tracker_hafm[name_log][1].update(loss_cur)

            # -------------------------------------- update time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # ------------------------------------ log information
            if iter % cfg.print_interval == 0:
                # print(data_path)
                log_str = f"[{epoch:2d}][{iter:5d}/{len(data_loader):5d}] " \
                          f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) " \
                          f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "

                for name_log, (_, tracker) in metric_tracker.items():
                    log_str += f"{name_log}: {tracker.val:.4f} ({tracker.avg:.4f}) "
                for name_log, (_, tracker) in metric_tracker_hafm.items():
                    log_str += f"{name_log}: {tracker.val:.4f} ({tracker.avg:.4f}) "
                logger.info(log_str)

                print(f"[{model_name}-> {epoch:2d}][{iter:5d}/{len(data_loader):5d}] "
                      f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                      f"Loss: {losses.val:.4f} ({losses.avg:.4f}) ")
                logger.info('-------------------------------------')

        lr_scheduler.step()
        writer.add_scalar('Loss/hafm', metric_tracker_hafm['LOSS_HAFM'][1].avg, epoch)
        if cfg.stage >= 2:
            writer.add_scalar('Loss/ce', metric_tracker['Classify_instance'][1].avg, epoch)
            writer.add_scalar('Loss/juncs', metric_tracker['Regression_Juncs'][1].avg, epoch)
        if cfg.stage >= 3:
            writer.add_scalar('Loss/veri_hidden_pos',metric_tracker['LOSS_Hidden_pos'][1].avg,epoch)
            writer.add_scalar('Loss/veri_hidden_neg', metric_tracker['LOSS_Hidden_neg'][1].avg,epoch)
        if cfg.stage == 4:
            writer.add_scalar('Loss/offset', metric_tracker['Offset_prediction'][1].avg, epoch)

        # log for one epoch
        logger.info('*' * 40)
        log_str = f"[{epoch:2d}] " \
                  f"Loss: {losses.avg:.4f} "
        for name_log, (_, tracker) in metric_tracker.items():
            log_str += f"{name_log}: {tracker.avg:.4f} "
        for name_log, (_, tracker) in metric_tracker_hafm.items():
            log_str += f"{name_log}: {tracker.avg:.4f} "
        logger.info(log_str)
        logger.info('*' * 40)
        # save checkpoint
        if cfg.save_model:
            if (epoch) % cfg.save_step == 0 or epoch >= 25:
                torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))
                torch.save(network.state_dict(), os.path.join(checkpoint_dir, "last_checkpoint.pt"))

if __name__ == '__main__':
    cfg = Set_Config(args)
    logger = Set_Logger(args, cfg)
    # ------------------------------------------ main
    if args.mode == 'train':
        train(cfg, logger)
    else:
        exit()