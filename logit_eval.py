import os
import sys
from operator import itemgetter

import sklearn
import sklearn.metrics
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def uncertainty_score(logit, top_k):
    score_list = []
    for idx in range(len(logit)):
        logit_i = logit[idx].data.cpu().numpy().tolist()
        indices, L_sorted = zip(*sorted(enumerate(logit_i), key=itemgetter(1), reverse=True))
        score_i = (top_k * L_sorted[0] - sum(L_sorted[1:top_k + 1])) / top_k
        score_list.append(score_i)
    return score_list

## general evaluation
def logit_eval(dataset, x_val, y_val, model, args):
    print("using logit diff evaluation ...")
    model.eval()
    corrects, avg_loss = 0, 0
    #print(model)
    y_pred = []
    y_truth = []
    logit_diff = []
    logit_var = []
    uncertainty_list = []
    represent_all = torch.FloatTensor()
    target_all = torch.LongTensor()
    val_iter = dataset.gen_minibatch(x_val, y_val, args.batch_size, args, shuffle=True)
    for batch in val_iter:
        feature, target = batch[0], batch[1]
        #feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit, represent = model(feature)
        #loss = F.cross_entropy(logit, target, size_average=False)

        #avg_loss += loss.data[0]
        y_pred_cur = (torch.max(logit, 1)[1].view(target.size()).data).tolist()
        y_truth_cur = target.data.tolist()

        y_pred += y_pred_cur
        y_truth += y_truth_cur
        # corrects += (torch.max(logit, 1)
        #              [1].view(target.size()).data == target.data).sum()
        top_k = args.logitev_topk
        if args.class_num <= top_k:
            print("Warning: parameter logitev-topk is larger than the class number")
            top_k = args.class_num - 1
        uncertainty_cur = uncertainty_score(logit, top_k=top_k)
        uncertainty_list += uncertainty_cur
        # logit_var_cur = np.var(logit.cpu().data.numpy(), axis=1).tolist()
        # logit_var += logit_var_cur
        # logit_diff_cur = (logit[:, 1] - logit[:, 0]).data.tolist()
        # logit_diff += logit_diff_cur
        represent_all = torch.cat([represent_all, represent.data.cpu()], 0)
        target_all = torch.cat([target_all, target.data.cpu()], 0)

        # print("\n=== logit ===")
        # print(logit)
        # print("=== prediction ===")
        # print(y_pred_cur)
        # print("=== truth ===")
        # print(y_truth_cur)

    # apply idk ratio to filter out the uncertain instances.
    if args.use_idk:
        # logit_diff_abs = [abs(x) for x in logit_diff]
        # indices, L_sorted = zip(*sorted(enumerate(logit_diff_abs), key=itemgetter(1), reverse=True))
        # indices, L_sorted = zip(*sorted(enumerate(logit_var), key=itemgetter(1), reverse=True))
        indices, L_sorted = zip(*sorted(enumerate(uncertainty_list), key=itemgetter(1), reverse=True))

        idk_list = np.arange(0, 0.45, 0.05)
        for idk_ratio in idk_list:
            #print("=== idk_ratio: ", idk_ratio, " ===")
            test_num = int(len(L_sorted) * (1 - idk_ratio))
            indices_cur = list(indices[:test_num])
            y_truth_cur = [y_truth[i] for i in indices_cur]
            y_pred_cur = [y_pred[i] for i in indices_cur]
            f1_score = show_results(dataset, y_truth_cur, y_pred_cur, represent_all, target_all)

    else:
        f1_score = show_results(dataset, y_truth, y_pred, represent_all, target_all)

    return f1_score


def show_results(dataset, y_truth, y_pred, represent=None, target=None):

    class_num = dataset.get_class_num()

    if class_num == 2:
        accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
        f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, pos_label=1))
        prec_score = (sklearn.metrics.precision_score(y_truth, y_pred, pos_label=1))
        recall_score = (sklearn.metrics.recall_score(y_truth, y_pred, pos_label=1))
        confusion_mat = (sklearn.metrics.confusion_matrix(y_truth, y_pred))

        print(accuracy_score, f1_score, prec_score, recall_score, sep='\t')
        print(confusion_mat)
        return f1_score

    elif class_num > 2:
        accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
        micro_f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, average='micro'))
        macro_f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, average='macro'))
        print(accuracy_score, micro_f1_score, macro_f1_score, sep='\t')
        return micro_f1_score

    # if represent is not None:
    #     loss_intra, loss_inter = eval_metric(represent, target)
    #     print(loss_intra, loss_inter)

    # print('\nEvaluation - acc: {:.4f} f1: {:.4f} precision: {:.4f} recall: {:.4f}\n'.format(accuracy_score,
    #                                                                              f1_score,
    #                                                                              prec_score,
    #                                                                              recall_score))
