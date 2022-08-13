import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import os.path as osp

def TPFP(lines_dt, lines_gt, threshold):
    lines_dt = lines_dt.reshape(-1,2,3)[:,:,::-1]
    lines_gt = lines_gt.reshape(-1,2,3)[:,:,::-1]
    diff = ((lines_dt[:, None, :, None] - lines_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff,1)
    dist = np.min(diff,1)
    hit = np.zeros(len(lines_gt), np.bool)
    tp = np.zeros(len(lines_dt), np.float)
    fp = np.zeros(len(lines_dt),np.float)

    for i in range(lines_dt.shape[0]):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp

def AP(tp, fp):
    recall = tp
    precision = tp/np.maximum(tp+fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))



    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    return ap

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Structural AP Evaluation')
    argparser.add_argument('--path',dest='path',type=str,required=True)
    argparser.add_argument('-t','--threshold', dest='threshold', type=float, default=0.01)

    args = argparser.parse_args()

    result_path = args.path
    with open(result_path, 'r') as _res:
        result_list = json.load(_res)

    tp_list, fp_list, scores_list = [], [], []
    tp_list_v, fp_list_v, scores_list_v = [], [], []
    tp_list_h, fp_list_h, scores_list_h = [], [], []
    
    n_gt = 0
    n_gt_v = 0
    n_gt_h = 0
    k = 0
    for res in result_list:
        if len(res['lines_hidden_pred'])==0:
            continue
        k += 1
        juncs_cc_pred = np.array(res['junctions_cc_refined'])
        juncs_cc_gt = np.array(res['junctions_cc_gt'])
        idx_lines_for_juncs_visi = np.array(res['idx_lines_for_junctions_visi'])
        idx_lines_for_juncs_hidden = np.array(res['idx_lines_for_junctions_hidden'])
        edges_positive_visi = np.array(res['edges_positive_visi'])
        edges_positive_hidden = np.array(res['edges_positive_hidden'])
        
        lines_pred_h = juncs_cc_pred[idx_lines_for_juncs_hidden].reshape((-1,6))
        scores_h = np.array(res['lines_hidden_score'])
        lines_gt_h = juncs_cc_gt[edges_positive_hidden]
        
        lines_pred_v = juncs_cc_pred[idx_lines_for_juncs_visi].reshape((-1, 6))
        scores_v = np.array(res['lines_visible_score'])
        lines_gt_v = juncs_cc_gt[edges_positive_visi]
        
        idx_lines_for_juncs = np.concatenate([idx_lines_for_juncs_visi,idx_lines_for_juncs_hidden],axis=0)
        edges_positive = np.concatenate([edges_positive_visi,edges_positive_hidden],axis=0)
        scores_v = np.array(res['lines_visible_score'],dtype=np.float)
        scores_h = np.array(res['lines_hidden_score'],dtype=np.float)
        scores = np.concatenate([scores_v,scores_h],axis=0)
        lines_pred = juncs_cc_pred[idx_lines_for_juncs]
        lines_gt = juncs_cc_gt[edges_positive]

        sort_idx = np.argsort(-scores)
        lines_pred = lines_pred[sort_idx].reshape((-1,6))
        scores = scores[sort_idx]
        tp, fp = TPFP(lines_pred, lines_gt, args.threshold)
        n_gt += lines_gt.shape[0]
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(scores)

        sort_idx_v = np.argsort(-scores_v)
        lines_pred_v = lines_pred_v[sort_idx_v].reshape((-1,6))
        scores_v = scores_v[sort_idx_v]
        tp_v, fp_v = TPFP(lines_pred_v, lines_gt_v, args.threshold)
        n_gt_v += lines_gt_v.shape[0]
        tp_list_v.append(tp_v)
        fp_list_v.append(fp_v)
        scores_list_v.append(scores_v)

        sort_idx_h = np.argsort(-scores_h)
        lines_pred_h = lines_pred_h[sort_idx_h].reshape((-1,6))
        scores_h = scores_h[sort_idx_h]
        tp_h, fp_h = TPFP(lines_pred_h, lines_gt_h, args.threshold)
        n_gt_h += lines_gt_h.shape[0]
        tp_list_h.append(tp_h)
        fp_list_h.append(fp_h)
        scores_list_h.append(scores_h)


    tp_list = np.concatenate(tp_list)
    fp_list = np.concatenate(fp_list)
    scores_list = np.concatenate(scores_list)
    idx = np.argsort(scores_list)[::-1]
    tp = np.cumsum(tp_list[idx]) / n_gt
    fp = np.cumsum(fp_list[idx]) / n_gt
    rcs = tp
    pcs = tp / np.maximum(tp + fp, 1e-9)
    sAP = AP(tp, fp) * 100
    sAP_string = 'sAP{} = {:.1f}'.format(args.threshold, sAP)
    print(sAP_string)

    tp_list_v = np.concatenate(tp_list_v)
    fp_list_v = np.concatenate(fp_list_v)
    scores_list_v = np.concatenate(scores_list_v)
    idx_v = np.argsort(scores_list_v)[::-1]
    tp_v = np.cumsum(tp_list_v[idx_v]) / n_gt_v
    fp_v = np.cumsum(fp_list_v[idx_v]) / n_gt_v
    rcs_v = tp_v
    pcs_v = tp_v / np.maximum(tp_v + fp_v, 1e-9)
    sAP_v = AP(tp_v, fp_v) * 100
    sAP_string_v = 'sAP_v{} = {:.1f}'.format(args.threshold, sAP_v)
    print(sAP_string_v)

    tp_list_h = np.concatenate(tp_list_h)
    fp_list_h = np.concatenate(fp_list_h)
    scores_list_h = np.concatenate(scores_list_h)
    idx_h = np.argsort(scores_list_h)[::-1]
    tp_h = np.cumsum(tp_list_h[idx_h]) / n_gt_h
    fp_h = np.cumsum(fp_list_h[idx_h]) / n_gt_h
    rcs_h = tp_h
    pcs_h = tp_h / np.maximum(tp_h + fp_h, 1e-9)
    sAP_h = AP(tp_h, fp_h) * 100
    sAP_string_h = 'sAP_h{} = {:.1f}'.format(args.threshold, sAP_h)
    print(sAP_string_h)

    #import pdb;pdb.set_trace()
    results_pc = np.load('1.npy')
    rcs_pc = results_pc[0]
    pcs_pc = results_pc[1]
    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color=[0, 0.5, 0], alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4, fontsize=10)

    plt.rc('legend', fontsize=10)
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    v, = plt.plot(rcs_v, pcs_v, 'm-')
    h, = plt.plot(rcs_h, pcs_h, 'b-')
    all, = plt.plot(rcs,pcs, 'r-')
    pc, =  plt.plot(rcs_pc,pcs_pc, 'c-')
    plt.title('sAP Evaluation for 3D Lines (threshold=0.07)')
    plt.legend([v,h,all,pc],['Visible lines(Ours)', 'Hidden lines(Ours)','All lines(Ours)','All lines(PC2WF)'],loc='lower right')
    plt.savefig('sAP{}.pdf'.format(args.threshold))

    