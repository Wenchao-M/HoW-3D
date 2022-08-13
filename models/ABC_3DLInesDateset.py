import os
import os.path as osp
from torch.utils import data
import numpy as np
import cv2
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torch
import json
from skimage import io

class ABC_3DLineDataset(data.Dataset):
    def __init__(self, subset='train', transform=None, root_dir=None, reslution=[256,256]):
        assert subset in ['train', 'test']
        self.subset = subset
        self.transform = transform
        self.root_dir = root_dir
        self.reslution = reslution
        if self.subset == 'train':
            self.data_list = [line.strip() for line in open(osp.join(self.root_dir,"train_lst.txt")).readlines()]
        else:
            self.data_list = [line.strip() for line in open(osp.join(self.root_dir, "test_lst.txt")).readlines()]
        self.reprojection_matrix = np.linalg.pinv(np.mat([(1.5625, 0.0000,  0.0000,  0.0000),
            (0.0000, 1.5625,  0.0000,  0.0000),
            (0.0000, 0.0000, -1.0020, -0.2002),
            (0.0000, 0.0000, -1.0000,  0.0000)]))

    def __getitem__(self, index):
        if self.subset == "test":
            data_path = self.data_list[index]
            json_path,image_path = data_path.split(" ")
            image = io.imread(osp.join(self.root_dir, image_path)).astype(float)[:, :, :3]/255.0
            ori_h, ori_w = image.shape[:2]
            image = cv2.resize(image,(self.reslution))
            if self.transform is not None:
                image = self.transform(image)

            with open(osp.join(self.root_dir,json_path)) as _:
                data = json.load(_)

            scale_x = self.reslution[0]/ori_w
            scale_y = self.reslution[1]/ori_h
            fname = image_path.split('/')[-1]
            juncs_2D = torch.tensor(data['junctions']).float()
            juncs_3D = torch.tensor(data['junctions_cc']).float()
            juncs_2D[:,0] *= scale_x
            juncs_2D[:,1] *= scale_y
            juncs_label = torch.tensor(data['junction_label']).long()
            edges_visi = torch.tensor(data['lines_positive_visible']).long()
            edges_hidden = torch.tensor(data['lines_positive_hidden']).long()
            edges_negative = torch.tensor(data['lines_negative']).long()
            lines_visi_2D = juncs_2D[edges_visi].reshape((-1,4))
            lines_pad_visi_2D = torch.zeros((40,4),dtype=torch.float)
            juncs_hidden = juncs_2D[juncs_label==2]
            num_lines = lines_visi_2D.shape[0]
            if num_lines != 0:
                lines_pad_visi_2D[:num_lines,:] = lines_visi_2D

            sample = {'lines_visi_2D': lines_pad_visi_2D,
                      'num_lines': num_lines,
                      'fname':fname,
                      'image': image,
                      'height': self.reslution[0],
                      'width': self.reslution[1]
            }

            label = torch.zeros(juncs_hidden.shape[0],dtype=torch.long)
            target = {
                'labels':label,
                'junctions_2D': juncs_2D,
                'junctions_cc': juncs_3D,
                'junction_hidden': juncs_hidden/self.reslution[0],
                'junction_label': juncs_label,
                'edges_positive_visible': edges_visi,
                'edges_positive_hidden':edges_hidden,
                'edges_negative':edges_negative,
                'height': self.reslution[0],
                'width': self.reslution[1]
            }
        else:
            idx_ = index % len(self.data_list)
            reminder = index // len(self.data_list)
            data_path = self.data_list[idx_]
            json_path, image_path = data_path.split(" ")
            image = io.imread(osp.join(self.root_dir, image_path)).astype(float)[:, :, :3] / 255.0
            ori_h, ori_w = image.shape[:2]
            with open(osp.join(self.root_dir, json_path)) as _:
                data = json.load(_)
            scale_x = self.reslution[0] / ori_w
            scale_y = self.reslution[1] / ori_h
            fname = image_path.split('/')[-1]
            juncs_2D = torch.tensor(data['junctions']).float()
            juncs_3D = np.array(data['junctions_cc'],dtype=np.float)
            juncs_2D[:, 0] *= scale_x
            juncs_2D[:, 1] *= scale_y
            juncs_label = torch.tensor(data['junction_label']).long()
            edges_visi = torch.tensor(data['lines_positive_visible']).long()
            edges_hidden = torch.tensor(data['lines_positive_hidden']).long()
            edges_negative = torch.tensor(data['lines_negative']).long()
            js = np.array(data['junctions'], dtype=np.float)
            dp = np.array(data['junctions_cc'], dtype=np.float)[:, 2]
            if reminder == 1:
                image = image[:, ::-1, :]
                juncs_2D[:, 0] = 256 - juncs_2D[:, 0]
                js[:,0] = 512 - js[:, 0]
                juncs_3D = self.reprojection(js,dp)

            elif reminder == 2:
                # image = F.vflip(image)
                image = image[::-1, :, :]
                juncs_2D[:,1] = 256 - juncs_2D[:, 1]
                js[:, 1] = 512 - js[:, 1]
                juncs_3D = self.reprojection(js, dp)

            elif reminder == 3:
                # image = F.vflip(F.hflip(image))
                image = image[::-1, ::-1, :]
                juncs_2D[:, 0] = 256 - juncs_2D[:, 0]
                juncs_2D[:, 1] = 256 - juncs_2D[:, 1]
                js[:,0] = 512 - js[:,0]
                js[:,1] = 512 - js[:,1]
                juncs_3D = self.reprojection(js,dp)
            else:
                pass

            juncs_3D = torch.tensor(juncs_3D,dtype=torch.float)
            lines_visi_2D = juncs_2D[edges_visi].reshape((-1, 4))
            lines_pad_visi_2D = torch.zeros((40, 4), dtype=torch.float)
            juncs_hidden = juncs_2D[juncs_label == 2]
            juncs_hidden_3D = juncs_3D[juncs_label==2]
            num_lines = lines_visi_2D.shape[0]
            if num_lines != 0:
                lines_pad_visi_2D[:num_lines, :] = lines_visi_2D

            image = cv2.resize(image, (self.reslution))
            if self.transform is not None:
                image = self.transform(image)
            sample = {'lines_visi_2D': lines_pad_visi_2D,
                      'num_lines': num_lines,
                      'fname': fname,
                      'image': image,
                      'height': self.reslution[0],
                      'width': self.reslution[1]
                      }

            label = torch.zeros(juncs_hidden.shape[0], dtype=torch.long)
            target = {
                'labels': label,
                'junctions_2D': juncs_2D,
                'junctions_cc': juncs_3D,
                'junction_hidden': juncs_hidden / self.reslution[0],
                'juncs_hidden_3D': juncs_hidden_3D,
                'junction_label': juncs_label,
                'edges_positive_visible': edges_visi,
                'edges_positive_hidden': edges_hidden,
                'edges_negative': edges_negative,
                'height': self.reslution[0],
                'width': self.reslution[1]
            }

        return sample,target

    def reprojection(self,juncs, depth):
        juncs[:, 0] = ((juncs[:, 0] / (512 - 1)) * 2 - 1) * depth
        juncs[:, 1] = ((juncs[:, 1] / (512 - 1)) * (-2) + 1) * depth
        juns_new = np.concatenate([juncs, depth[:, None], np.ones((juncs.shape[0], 1), dtype=np.float)],
                                  axis=-1)
        juncs_t = np.dot(self.reprojection_matrix, juns_new.T).T
        juncs_t = np.concatenate([juncs_t[:,:2] * -1,depth[:,None]],axis=-1)

        return juncs_t

    def __len__(self):
        if self.subset == 'train':
            return len(self.data_list)* 4

        else:
            return len(self.data_list)

def build_dataset(subset='train', transform=None, root_dir='data'):

    dataset = ABC_3DLineDataset(subset,transform,root_dir)

    return dataset

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
