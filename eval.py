import os
import os.path as osp
import time
import numpy as np
import torch
from torch.utils import data
import json
import torchvision.transforms as tf
from utils.utils import Set_Config, Set_Logger, Set_Ckpt_Code_Debug_Dir
from models.DSG_Model import DSG_Model
from models.ABC_3DLInesDateset import build_dataset,collate_fn
import matplotlib
matplotlib.use('Agg')
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--mode', default='train', type=str,
                    help='train / eval')
parser.add_argument('--backbone', default='hrnet', type=str,
                    help='only support hrnet now')
parser.add_argument('--cfg_path', default='configs/config_planeTR_eval_s1.yaml', type=str,
                    help='full path of the config file')
args = parser.parse_args()
NUM_GPUS = torch.cuda.device_count()

torch.backends.cudnn.benchmark = True


def load_dataset(cfg, args):

    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.05822601, 0.05822601, 0.05822601], [0.16618816, 0.16618816, 0.16618816])
    ])
    assert NUM_GPUS > 0

    dataset = build_dataset('test',transforms)
    loaders = data.DataLoader(
        dataset,
        batch_size=1, shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    data_sampler = None

    return loaders, data_sampler



def test(cfg, logger):
    logger.info('*' * 40)
    localtime = time.asctime(time.localtime(time.time()))
    logger.info(localtime)
    logger.info('start training......')
    logger.info('*' * 40)

    model_name = (cfg.save_path).split('/')[-1]

    # set random seed

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build network
    network = DSG_Model(cfg)
    network = network.to(device)

    # load pretrained weights if existed
    if not (cfg.resume_dir == 'None'):
        print('loading checkpoint from ' + cfg.resume_dir)
        loc = 'cuda:{}'.format(args.local_rank)
        model_dict = network.state_dict()
        pretrained_dict = torch.load(cfg.resume_dir, map_location=loc)
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)

    data_loader, data_sampler = load_dataset(cfg, args)
    network.eval()
    results = []
    for iter, (sample,targets) in tqdm(enumerate(data_loader)):
        image = sample['image'].to(device).float()  # b, 3, h, w
        result = network(image)
        result['junctions_gt'] = (targets[0]['junctions_2D']/2).tolist()
        result['junctions_label_gt'] = targets[0]['junction_label'].tolist()
        result['edges_positive_visi'] = targets[0]['edges_positive_visible'].tolist()
        result['edges_positive_hidden'] = targets[0]['edges_positive_hidden'].tolist()
        result['junctions_cc_gt'] = targets[0]['junctions_cc'].tolist()
        result['fname'] = sample['fname'][0]
        results.append(result)
    if not osp.isdir(cfg.save_path):
        os.mkdir(cfg.save_path)
    with open(osp.join(cfg.save_path,'results.json'),'w') as _:
        json.dump(results,_)


if __name__ == '__main__':
    cfg = Set_Config(args)

    # ------------------------------------------ set logger
    logger = Set_Logger(args, cfg)

    # ------------------------------------------ main
    if args.mode == 'train':
        test(cfg, logger)
    else:
        exit()