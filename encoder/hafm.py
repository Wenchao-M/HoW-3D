import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

import _C



class HAFMencoder(object):
    def __init__(self, cfg):
        self.dis_th = cfg.ENCODER.DIS_TH
        self.ang_th = cfg.ENCODER.ANG_TH
        self.num_static_pos_lines = cfg.ENCODER.NUM_STATIC_POS_LINES
        self.num_static_neg_lines = cfg.ENCODER.NUM_STATIC_NEG_LINES
    def __call__(self,annotations):
        targets = []
        metas   = []
        for ann in annotations:
            t,m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        
        return default_collate(targets),metas

    def gen_hafm(self,lmap,height,width):
        dismap = torch.sqrt(lmap[0] ** 2 + lmap[1] ** 2)[None]

        def _normalize(inp):
            mag = torch.sqrt(inp[0] * inp[0] + inp[1] * inp[1])
            return inp / (mag + 1e-6)

        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])

        md_ = md_map.reshape(2, -1).t()
        st_ = st_map.reshape(2, -1).t()
        ed_ = ed_map.reshape(2, -1).t()
        Rt = torch.cat(
            (torch.cat((md_[:, None, None, 0], md_[:, None, None, 1]), dim=2),
             torch.cat((-md_[:, None, None, 1], md_[:, None, None, 0]), dim=2)), dim=1)

        Rtst_ = torch.matmul(Rt, st_[:, :, None]).squeeze(-1).t()
        Rted_ = torch.matmul(Rt, ed_[:, :, None]).squeeze(-1).t()
        swap_mask = (Rtst_[1] < 0) * (Rted_[1] > 0)
        pos_ = Rtst_.clone()
        neg_ = Rted_.clone()
        temp = pos_[:, swap_mask]
        pos_[:, swap_mask] = neg_[:, swap_mask]
        neg_[:, swap_mask] = temp

        pos_[0] = pos_[0].clamp(min=1e-9)
        pos_[1] = pos_[1].clamp(min=1e-9)
        neg_[0] = neg_[0].clamp(min=1e-9)
        neg_[1] = neg_[1].clamp(max=-1e-9)

        mask = ((pos_[1] > self.ang_th) * (neg_[1] < -self.ang_th) * (dismap.view(-1) <= self.dis_th)).float()

        pos_map = pos_.reshape(-1, height, width)
        neg_map = neg_.reshape(-1, height, width)

        md_angle = torch.atan2(md_map[1], md_map[0])
        pos_angle = torch.atan2(pos_map[1], pos_map[0])
        neg_angle = torch.atan2(neg_map[1], neg_map[0])

        pos_angle_n = pos_angle / (np.pi / 2)
        neg_angle_n = -neg_angle / (np.pi / 2)
        md_angle_n = md_angle / (np.pi * 2) + 0.5
        mask = mask.reshape(height, width)

        hafm_ang = torch.cat((md_angle_n[None], pos_angle_n[None], neg_angle_n[None],), dim=0)
        hafm_dis = dismap.clamp(max=self.dis_th) / self.dis_th
        mask = mask[None]

        return [hafm_ang,hafm_dis,mask]

    def adjacent_matrix(self,juncs,juncs_vh,juncs_,edges, device):

        juncs_new = torch.cat((juncs_vh,juncs_),dim=0)
        n = juncs_new.size(0)
        mat = torch.zeros(n+2,n+2,dtype=torch.bool,device=device)
        cost_,match_= torch.sum((juncs - juncs_new[:, None]) ** 2, dim=-1).min(0)
        match_[cost_ > 0.01] = n+1

        if edges.size(0)>0:
            mat[match_[edges[:,0]], match_[edges[:,1]]] = 1
            mat[match_[edges[:,1]], match_[edges[:,0]]] = 1
        return mat

    def _gen_jmap(self,junctions,juncs_z,height,width):

        # junctions[:,0] = junctions[:,0].clamp(0,width-1)
        # junctions[:,1] = junctions[:,1].clamp(0,height-1)
        device = junctions.device
        jmap = torch.zeros((height, width), device=device)
        joff = torch.zeros((2, height, width), device=device, dtype=torch.float32)
        j_depth = torch.zeros((height,width),device=device)
        xint, yint = junctions[:, 0].long(), junctions[:, 1].long()
        off_x = junctions[:, 0] - xint.float() - 0.5
        off_y = junctions[:, 1] - yint.float() - 0.5
        jmap[yint, xint] = 1
        j_depth[yint,xint] = juncs_z
        joff[0, yint, xint] = off_x
        joff[1, yint, xint] = off_y

        return jmap,joff, j_depth



    def _process_per_image(self,ann):

        junctions = ann['junctions_2D']/2
        device = junctions.device
        height, width = int(ann['height']/2), int(ann['width']/2)
        junction_label = ann['junction_label']
        juncs_z = ann["junctions_cc"][:,2]

        junctions_v = junctions[junction_label==1]
        juncs_z_v = juncs_z[junction_label==1]
        junctions_h = junctions[junction_label==2]
        juncs_z_h = juncs_z[junction_label==2]
        junctions_vh = junctions[junction_label==3]
        juncs_z_vh = juncs_z[junction_label==3]

        jmap_v, joff_v, jdepth_v = self._gen_jmap(junctions_v,juncs_z_v, height,width)
        jmap_h, joff_h, jdepth_h = self._gen_jmap(junctions_h,juncs_z_h, height,width)
        jmap_vh,joff_vh, jdepth_vh = self._gen_jmap(junctions_vh,juncs_z_vh,height,width)

        edges_positive_visible = ann['edges_positive_visible']
        edges_positive_hidden = ann['edges_positive_hidden']
        edges_negative = ann['edges_negative']

        lines_visi = torch.cat((junctions[edges_positive_visible[:,0]], junctions[edges_positive_visible[:,1]]),dim=-1)
        lines_hidden = torch.cat((junctions[edges_positive_hidden[:, 0]], junctions[edges_positive_hidden[:, 1]]), dim=-1)
        lines_neg = torch.cat((junctions[edges_negative[:2000, 0]], junctions[edges_negative[:2000, 1]]), dim=-1)
        lmap_visi, _, _ = _C.encodels(lines_visi,height,width,height,width,lines_visi.size(0))

        pos_mat_visi = self.adjacent_matrix(junctions, junctions_vh, junctions_v, edges_positive_visible, device)
        pos_mat_hidd = self.adjacent_matrix(junctions, junctions_vh, junctions_h, edges_positive_hidden, device)
        lpos_visi = np.random.permutation(lines_visi.cpu().numpy())[:self.num_static_pos_lines]
        lpos_hidden = np.random.permutation(lines_hidden.cpu().numpy())[:self.num_static_pos_lines]
        lneg = np.random.permutation(lines_neg.cpu().numpy())[:self.num_static_neg_lines]
        lpos_visi = torch.from_numpy(lpos_visi).to(device)
        lpos_hidden = torch.from_numpy(lpos_hidden).to(device)
        lneg = torch.from_numpy(lneg).to(device)

        lpre_visi = torch.cat((lpos_visi, lneg), dim=0)
        lpre_hidden = torch.cat((lpos_hidden, lneg), dim=0)
        _swap = (torch.rand(lpre_visi.size(0)) > 0.5).to(device)
        lpre_visi[_swap] = lpre_visi[_swap][:, [2, 3, 0, 1]]
        _swap = (torch.rand(lpre_hidden.size(0)) > 0.5).to(device)
        lpre_hidden[_swap] = lpre_hidden[_swap][:, [2, 3, 0, 1]]

        lpre_label_visi = torch.cat(
            [
                torch.ones(lpos_visi.size(0), device=device),
                torch.zeros(lneg.size(0), device=device)
            ])
        lpre_label_hidden = torch.cat(
            [
                torch.ones(lpos_hidden.size(0), device=device),
                torch.zeros(lneg.size(0), device=device)
            ])

        meta = {
            'junc': junctions,
            'junc_v': junctions_v,
            'junc_h': junctions_h,
            'junc_vh': junctions_vh,
            'Lpos_visi': pos_mat_visi,
            'Lpos_hidden': pos_mat_hidd,
            'lpre_visi': lpre_visi,
            'lpre_label_visi': lpre_label_visi,
            'lpre_hidden': lpre_hidden,
            'lpre_label_hidden': lpre_label_hidden,
            'lines_visi': lines_visi,
            'lines_hidden': lines_hidden
        }

        hafm_visible = self.gen_hafm(lmap_visi,height,width)

        target = {'jloc_v':jmap_v[None],
                'joff_v':joff_v,
                'jdepth_v':jdepth_v[None],
                'jloc_h': jmap_h[None],
                'joff_h': joff_h,
                'jdepth_h': jdepth_h[None],
                'jloc_vh':jmap_vh[None],
                'joff_vh':joff_vh,
                'jdepth_vh':jdepth_vh[None],
                'md_visi': hafm_visible[0],
                'dis_visi': hafm_visible[1],
                'mask_visi': hafm_visible[2],
        }

        return target, meta