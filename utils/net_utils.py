import numpy as np
import torch
import torch.nn.functional as F

def gather_feature(id, feature):
    feature_id = id.unsqueeze_(2).long().expand(id.size(0),
                                                id.size(1),
                                                feature.size(2)).detach()
    cnn_out = torch.gather(feature, 1, feature_id).float()

    return cnn_out

def reprojection(reprojection_matrix, junctions,depth):
    #junctions: num_junctions,2
    #depth: num_junctions,1
    junctions[:, 0] = ((junctions[:, 0] * 4 / (512 - 1)) * 2 - 1) * depth[:,0]
    junctions[:, 1] = ((junctions[:, 1] * 4 / (512 - 1)) * (-2) + 1) * depth[:,0]
    pp = torch.ones((junctions.shape[0], 1), dtype=torch.float, device='cuda')
    junctions = torch.cat([junctions, depth, pp], dim=-1)
    junctions_cc = torch.mm(reprojection_matrix, junctions.T).T
    junctions_cc = torch.cat([junctions_cc[:, :2] * -1, depth], dim=-1)

    return junctions_cc
def get_lines_features(feat,lines, size_ori, n_pts=21):
    """
    :param feat: B, C, H, W
    :param lines: B, N, 4
    :param lines_3D B,N,6
    :return: B, C, N
    """
    ho, wo = size_ori
    b, c, hf, wf = feat.shape
    line_num = lines.shape[1]

    with torch.no_grad():
        scale_h = ho / hf
        scale_w = wo / wf
        scaled_lines = lines.clone()
        scaled_lines[:, :, 0] = scaled_lines[:, :, 0] / scale_w
        scaled_lines[:, :, 1] = scaled_lines[:, :, 1] / scale_h
        scaled_lines[:, :, 2] = scaled_lines[:, :, 2] / scale_w
        scaled_lines[:, :, 3] = scaled_lines[:, :, 3] / scale_h

        spts, epts = torch.split(scaled_lines, (2, 2), dim=-1)  # B, N, 2

        if n_pts > 2:
            delta_pts = (epts - spts) / (n_pts-1)  # B, N, 2
            delta_pts = delta_pts.unsqueeze(dim=2).expand(b, line_num, n_pts, 2)  # b, n, n_pts, 2
            steps = torch.linspace(0., n_pts-1, n_pts).view(1, 1, n_pts, 1).to(device=lines.device)

            spts_expand = spts.unsqueeze(dim=2).expand(b, line_num, n_pts, 2)  # b, n, n_pts, 2
            line_pts = spts_expand + delta_pts * steps  # b, n, n_pts, 2

        elif n_pts == 2:
            line_pts = torch.stack([spts, epts], dim=2)  # b, n, n_pts, 2
        elif n_pts == 1:
            line_pts = torch.cat((spts, epts), dim=1).unsqueeze(dim=2)
        line_pts[:, :, :, 0] = line_pts[:, :, :, 0] / float(wf-1) * 2. - 1.
        line_pts[:, :, :, 1] = line_pts[:, :, :, 1] / float(hf-1) * 2. - 1.

        line_pts = line_pts.detach()

    sample_feats = F.grid_sample(feat, line_pts)  # b, c, n, n_pts

    b, c, ln, pn = sample_feats.shape
    sample_feats = sample_feats.permute(0, 1, 3, 2).contiguous().view(b, -1, ln)


    return sample_feats

def interpolated_sum(cnns, coords, grids):
    X = coords[:, :, 0]
    Y = coords[:, :, 1]
    grid = grids[0]
    # x is the horizontal coordinate
    Xs = X * grid[1]
    X0 = torch.floor(Xs)
    X1 = X0 + 1
    Ys = Y * grid[0]
    Y0 = torch.floor(Ys)
    Y1 = Y0 + 1

    w_00 = (X1 - Xs) * (Y1 - Ys)
    w_01 = (X1 - Xs) * (Ys - Y0)
    w_10 = (Xs - X0) * (Y1 - Ys)
    w_11 = (Xs - X0) * (Ys - Y0)

    X0 = torch.clamp(X0, 0, grid[1] - 1)
    X1 = torch.clamp(X1, 0, grid[1] - 1)
    Y0 = torch.clamp(Y0, 0, grid[0] - 1)
    Y1 = torch.clamp(Y1, 0, grid[0] - 1)

    N1_id = X0 + Y0 * grid[1]
    N2_id = X0 + Y1 * grid[1]
    N3_id = X1 + Y0 * grid[1]
    N4_id = X1 + Y1 * grid[1]

    M_00 = gather_feature(N1_id, cnns[0])
    M_01 = gather_feature(N2_id, cnns[0])
    M_10 = gather_feature(N3_id, cnns[0])
    M_11 = gather_feature(N4_id, cnns[0])

    cnn_out = w_00.unsqueeze(2) * M_00 + \
              w_01.unsqueeze(2) * M_01 + \
              w_10.unsqueeze(2) * M_10 + \
              w_11.unsqueeze(2) * M_11

    return cnn_out

def gen_line_graph(max_nodes, ann, juncs_v_pred,z_v_pred, juncs_h_pred, z_h_pred, juncs_vh_pred, z_vh_pred, meta):
    juncs_label = ann['junction_label']
    juncs_gt = ann['junctions_2D'] / 2
    juncs_cc_gt = ann['junctions_cc']
    Lpos_visi = meta['Lpos_visi']
    Lpos_hidden = meta['Lpos_hidden']
    juncs_v_gt = juncs_gt[juncs_label==1]
    juncs_cc_v_gt = juncs_cc_gt[juncs_label==1]
    juncs_h_gt = juncs_gt[juncs_label==2]
    juncs_cc_h_gt = juncs_cc_gt[juncs_label==2]
    juncs_vh_gt = juncs_gt[juncs_label==3]
    juncs_cc_vh_gt = juncs_cc_gt[juncs_label==3]
    hi = (torch.arange(len(juncs_h_pred)) + len(juncs_vh_pred)).unsqueeze(1).expand(-1, (len(juncs_vh_pred) + len(juncs_h_pred)))
    vhi = (torch.arange(len(juncs_vh_pred) + len(juncs_h_pred))).unsqueeze(0).expand(len(juncs_h_pred), -1)
    ls = torch.stack([vhi, hi], dim=-1).flatten(0, 1)
    cost_h, match_h = torch.sum((juncs_h_pred - juncs_h_gt[:, None]) ** 2, dim=-1).min(0)
    match_h[cost_h > 2 * 2] = len(juncs_h_gt)
    juncs_h_pred[cost_h > 2*2] -= juncs_h_pred[cost_h > 2*2]
    z_h_pred[cost_h > 2*2] -= z_h_pred[cost_h > 2*2]
    juncs_cc_h_gt = torch.cat([juncs_cc_h_gt,torch.zeros((1,3),dtype=torch.float,device='cuda')],dim=0)
    juncs_h_pred_cc = juncs_cc_h_gt[match_h]
    cost_vh, match_vh = torch.sum((juncs_vh_pred - juncs_vh_gt[:, None]) ** 2, dim=-1).min(0)
    match_vh[cost_vh > 2 * 2] = len(juncs_vh_gt)
    juncs_cc_vh_gt = torch.cat([juncs_cc_vh_gt,torch.zeros((1,3),dtype=torch.float,device='cuda')],dim=0)
    juncs_vh_pred_cc = juncs_cc_vh_gt[match_vh]
    match_vh[cost_vh > 2 * 2] = len(juncs_vh_gt) + len(juncs_h_gt)
    match_hi = match_h.unsqueeze(1).expand(-1, match_h.shape[0] + match_vh.shape[0]) + len(juncs_vh_gt)
    match_vhi = torch.cat([match_vh, match_h + len(juncs_vh_gt)], dim=0).unsqueeze(0).expand(match_h.shape[0], -1)
    ls_match = torch.stack([match_hi, match_vhi], dim=-1).flatten(0, 1)
    label = Lpos_hidden[ls_match[:, 0], ls_match[:, 1]]
    idx = (ls[:, 0] < ls[:, 1])
    ls = ls[idx]
    label = label[idx]
    ls_positive_hidden = ls[label == 1]
    vi = (torch.arange(juncs_v_pred.shape[0] + juncs_vh_pred.shape[0])).unsqueeze(1).expand(-1, (juncs_vh_pred.shape[0] + juncs_v_pred.shape[0]))
    vhi = torch.arange(juncs_v_pred.shape[0]+juncs_vh_pred.shape[0]).unsqueeze(0).expand(juncs_v_pred.shape[0]+juncs_vh_pred.shape[0], -1)
    ls = torch.stack([vhi, vi], dim=-1).flatten(0, 1)
    cost_v, match_v = torch.sum((juncs_v_pred - juncs_v_gt[:, None]) ** 2, dim=-1).min(0)
    match_v[cost_v > 2 * 2] = len(juncs_v_gt)
    juncs_cc_v_gt = torch.cat([juncs_cc_v_gt,torch.zeros((1,3),dtype=torch.float,device='cuda')],dim=0)
    juncs_v_pred_cc = juncs_cc_v_gt[match_v]
    juncs_v_pred[cost_v > 2*2] -= juncs_v_pred[cost_v > 2*2]
    z_v_pred[cost_v > 2*2] -= z_v_pred[cost_v > 2*2]
    match_vh[cost_vh > 2 * 2] = len(juncs_vh_gt) + len(juncs_v_gt)
    match_vi = torch.cat([match_vh,match_v + juncs_vh_gt.shape[0]],dim=0).unsqueeze(1).expand(-1,match_v.shape[0] + match_vh.shape[0])
    match_vhi = torch.cat([match_vh, match_v + len(juncs_vh_gt)], dim=0).unsqueeze(0).expand(match_v.shape[0] + match_vh.shape[0],-1)
    ls_match = torch.stack([match_vhi, match_vi], dim=-1).flatten(0, 1)
    label = Lpos_visi[ls_match[:, 0], ls_match[:, 1]]
    idx = (ls[:, 0] < ls[:, 1])
    ls = ls[idx]
    label = label[idx]
    ls_positive_visi = ls[label == 1]
    juncs_pred = torch.cat([juncs_vh_pred,juncs_v_pred,juncs_h_pred],dim=0)
    depth_pred = torch.cat([z_vh_pred.unsqueeze(1),z_v_pred.unsqueeze(1),z_h_pred])
    juncs_cc = torch.cat([juncs_vh_pred_cc,juncs_v_pred_cc,juncs_h_pred_cc],dim=0)
    ls_positive_hidden[ls_positive_hidden>=juncs_vh_pred.shape[0]] = ls_positive_hidden[ls_positive_hidden>=juncs_vh_pred.shape[0]] + juncs_v_pred.shape[0]
    lines = torch.cat([ls_positive_visi,ls_positive_hidden],dim=0)
    num_juncs = juncs_pred.shape[0]
    juncs_pad = torch.zeros((max_nodes,2),dtype=torch.float,device='cuda')
    juncs_pad[:juncs_pred.shape[0]] = juncs_pred
    depth_pad = torch.zeros((max_nodes,1),dtype=torch.float,device='cuda')
    depth_pad[:depth_pred.shape[0]] = depth_pred
    juncs_cc_pad = torch.zeros((max_nodes,3),dtype=torch.float,device='cuda')
    juncs_cc_pad[:juncs_cc.shape[0]] = juncs_cc
    adj_matrix = torch.zeros((max_nodes,max_nodes),dtype=torch.float,device='cuda')
    adj_matrix[lines[:,0],lines[:,1]] = 1
    adj_matrix[lines[:,1],lines[:,0]] = 1
    adj_matrix = adj_matrix + torch.eye(max_nodes,dtype=torch.float,device='cuda')
    normalized_adj_matrix = adj_matrix / (torch.sum(adj_matrix, dim=1, keepdim=True) + 1e-6).repeat(1,adj_matrix.shape[-1])

    return juncs_pad,depth_pad, juncs_cc_pad, normalized_adj_matrix,num_juncs,ls_positive_visi,ls_positive_hidden

def proposal_lines(md_maps, dis_maps, scale=5.0):
    """
    :param md_maps: 3xhxw, the range should be (0,1) for every element
    :param dis_maps: 1xhxw
    :return:
    """
    device = md_maps.device
    height, width = md_maps.size(1), md_maps.size(2)
    _y = torch.arange(0, height, device=device).float()
    _x = torch.arange(0, width, device=device).float()

    y0, x0 = torch.meshgrid(_y, _x)
    md_ = (md_maps[0] - 0.5) * np.pi * 2
    st_ = md_maps[1] * np.pi / 2
    ed_ = -md_maps[2] * np.pi / 2

    cs_md = torch.cos(md_)
    ss_md = torch.sin(md_)

    cs_st = torch.cos(st_).clamp(min=1e-3)
    ss_st = torch.sin(st_).clamp(min=1e-3)

    cs_ed = torch.cos(ed_).clamp(min=1e-3)
    ss_ed = torch.sin(ed_).clamp(max=-1e-3)

    x_standard = torch.ones_like(cs_st)

    y_st = ss_st / cs_st
    y_ed = ss_ed / cs_ed

    x_st_rotated = (cs_md - ss_md * y_st) * dis_maps[0] * scale
    y_st_rotated = (ss_md + cs_md * y_st) * dis_maps[0] * scale

    x_ed_rotated = (cs_md - ss_md * y_ed) * dis_maps[0] * scale
    y_ed_rotated = (ss_md + cs_md * y_ed) * dis_maps[0] * scale

    x_st_final = (x_st_rotated + x0).clamp(min=0, max=width - 1)
    y_st_final = (y_st_rotated + y0).clamp(min=0, max=height - 1)

    x_ed_final = (x_ed_rotated + x0).clamp(min=0, max=width - 1)
    y_ed_final = (y_ed_rotated + y0).clamp(min=0, max=height - 1)

    lines = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final)).permute((1, 2, 0))

    return lines

def proposal_lines_new(md_maps, dis_maps, residual_maps, scale=5.0):
    device = md_maps.device
    sign_pad     = torch.tensor([-1,0,1],device=device,dtype=torch.float32).reshape(3,1,1)

    if residual_maps is None:
        dis_maps_new = dis_maps.repeat((1,1,1))
    else:
        dis_maps_new = dis_maps.repeat((3,1,1))+sign_pad*residual_maps.repeat((3,1,1))
    height, width = md_maps.size(1), md_maps.size(2)
    _y = torch.arange(0,height,device=device).float()
    _x = torch.arange(0,width, device=device).float()

    y0,x0 = torch.meshgrid(_y,_x)
    md_ = (md_maps[0]-0.5)*np.pi*2
    st_ = md_maps[1]*np.pi/2
    ed_ = -md_maps[2]*np.pi/2

    cs_md = torch.cos(md_)
    ss_md = torch.sin(md_)

    cs_st = torch.cos(st_).clamp(min=1e-3)
    ss_st = torch.sin(st_).clamp(min=1e-3)

    cs_ed = torch.cos(ed_).clamp(min=1e-3)
    ss_ed = torch.sin(ed_).clamp(max=-1e-3)

    y_st = ss_st/cs_st
    y_ed = ss_ed/cs_ed

    x_st_rotated = (cs_md-ss_md*y_st)[None]*dis_maps_new*scale
    y_st_rotated =  (ss_md + cs_md*y_st)[None]*dis_maps_new*scale

    x_ed_rotated =  (cs_md - ss_md*y_ed)[None]*dis_maps_new*scale
    y_ed_rotated = (ss_md + cs_md*y_ed)[None]*dis_maps_new*scale

    x_st_final = (x_st_rotated + x0[None]).clamp(min=0,max=width-1)
    y_st_final = (y_st_rotated + y0[None]).clamp(min=0,max=height-1)

    x_ed_final = (x_ed_rotated + x0[None]).clamp(min=0,max=width-1)
    y_ed_final = (y_ed_rotated + y0[None]).clamp(min=0,max=height-1)

    lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,3,0))

    return lines #, normals

