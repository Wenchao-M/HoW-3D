import argparse
import numpy as np
import json
import matplotlib.pyplot as plt


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

def TPFP(juncs_pred,juncs_gt,threshold):

    dis = np.sum((juncs_gt - juncs_pred[:,None])**2, axis=-1)
    choice = np.argmin(dis,1)
    distance = np.min(dis,1)

    hit = np.zeros(len(juncs_gt), np.bool)
    tp = np.zeros(len(juncs_pred), np.float)
    fp = np.zeros(len(juncs_pred), np.float)

    for i in range(juncs_pred.shape[0]):
        if distance[i] < threshold*threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Structural AP Evaluation')
    argparser.add_argument('--path',dest='path',type=str,required=True)
    argparser.add_argument('-t','--threshold', dest='threshold', type=float, default=0.05)

    args = argparser.parse_args()

    result_path = args.path
    with open(result_path, 'r') as _res:
        result_list = json.load(_res)

    
    tp_list, fp_list, scores_list,error_list = [], [], [], []
    tp_list_v, fp_list_v, scores_list_v,error_list_v = [], [], [], []
    tp_list_h, fp_list_h, scores_list_h,error_list_h = [], [], [], []
    tp_list_vh, fp_list_vh, scores_list_vh,error_list_vh = [], [], [], []
    n_gt = 0
    n_gt_v = 0
    n_gt_h = 0
    n_gt_vh = 0
    k = 0
    for res in result_list:
        juncs_gt = np.array(res['junctions_cc_gt'],dtype=np.float)
        juncs_label = np.array(res['junctions_label_gt'],dtype=np.int)
        juncs_pred = np.array(res['junctions_cc_refined'],dtype=np.float)
        juncs_label_pred = np.array(res['junctions_label_pred'],dtype=np.float)
        juncs_score = np.array(res['junctions_score'],dtype=np.float)
        #import pdb;pdb.set_trace()
        juncs_gt_v = juncs_gt[juncs_label==1]
        juncs_pred_v = juncs_pred[juncs_label_pred==1]
        juncs_score_v = juncs_score[juncs_label_pred==1]
        juncs_gt_h = juncs_gt[juncs_label==2]
        juncs_pred_h = juncs_pred[juncs_label_pred==2]
        juncs_score_h = juncs_score[juncs_label_pred==2]
        juncs_gt_vh = juncs_gt[juncs_label==3]
        juncs_pred_vh = juncs_pred[juncs_label_pred==3]
        juncs_score_vh = juncs_score[juncs_label_pred==3]

        sort_idx = np.argsort(-juncs_score)
        sort_idx_v = np.argsort(-juncs_score_v)
        sort_idx_h = np.argsort(-juncs_score_h)
        sort_idx_vh = np.argsort(-juncs_score_vh)

        scores = juncs_score[sort_idx]
        scores_v = juncs_score_v[sort_idx_v]
        scores_h = juncs_score_h[sort_idx_h]
        scores_vh = juncs_score[sort_idx_vh]

        juncs_pred = juncs_pred[sort_idx]
        juncs_pred_v = juncs_pred_v[sort_idx_v]
        juncs_pred_h = juncs_pred_h[sort_idx_h]
        juncs_pred_vh = juncs_pred_vh[sort_idx_vh]
        tp, fp = TPFP(juncs_pred, juncs_gt,args.threshold)
        n_gt += juncs_gt.shape[0]
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(scores)

        tp_v, fp_v = TPFP(juncs_pred_v, juncs_gt_v,args.threshold)
        n_gt_v += juncs_gt_v.shape[0]
        tp_list_v.append(tp_v)
        fp_list_v.append(fp_v)
        scores_list_v.append(scores_v)

        tp_h, fp_h = TPFP(juncs_pred_h, juncs_gt_h,args.threshold)
        n_gt_h += juncs_gt_h.shape[0]
        tp_list_h.append(tp_h)
        fp_list_h.append(fp_h)
        scores_list_h.append(scores_h)

        tp_vh, fp_vh = TPFP(juncs_pred_vh, juncs_gt_vh,args.threshold)
        n_gt_vh += juncs_gt_vh.shape[0]
        tp_list_vh.append(tp_vh)
        fp_list_vh.append(fp_vh)
        scores_list_vh.append(scores_vh)


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

    tp_list_vh = np.concatenate(tp_list_vh)
    fp_list_vh = np.concatenate(fp_list_vh)
    scores_list_vh = np.concatenate(scores_list_vh)
    idx_vh = np.argsort(scores_list_vh)[::-1]
    tp_vh = np.cumsum(tp_list_vh[idx_vh]) / n_gt_vh
    fp_vh = np.cumsum(fp_list_vh[idx_vh]) / n_gt_vh
    rcs_vh = tp_vh
    pcs_vh = tp_vh / np.maximum(tp_vh + fp_vh, 1e-9)
    sAP_vh = AP(tp_vh, fp_vh) * 100
    sAP_string_vh = 'sAP_vh{} = {:.1f}'.format(args.threshold, sAP_vh)
    print(sAP_string_vh)

    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color=[0, 0.5, 0], alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4, fontsize=10)

    results_pc = np.load('2.npy')
    rcs_pc = results_pc[0]
    pcs_pc = results_pc[1]
    plt.rc('legend', fontsize=10)
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.plot(rcs, pcs, 'r-', label='All junctions(Ours)')
    plt.plot(rcs_v, pcs_v, 'm-',label='Visible junctions(Ours)')
    plt.plot(rcs_h, pcs_h, 'b-',label='Hidden junctions(Ours)')
    plt.plot(rcs_vh, pcs_vh,'c-',label='Fleeting junctions(Ours)')
    plt.plot(rcs_pc, pcs_pc,'y-',label='All junctions(PC2WF)')
    plt.legend(loc="lower right")
    plt.title("AP Evaluation for 3D Junctions (threshold=0.05)")
    #plt.show()
    plt.savefig('sAP_J_{}.pdf'.format(args.threshold))