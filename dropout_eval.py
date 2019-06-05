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
from eval import show_results

def uncertain_score(feature, model, n=100):
    model.train()

    probs = []
    for _ in range(n):
        logit, represent = model(feature)
        y_pred = (torch.max(logit, 1)[1].view(feature.size()[0]).data).tolist()
        probs += [y_pred]
    predictive_mean = np.mean(probs, axis=0)
    predictive_variance = np.var(probs, axis=0)
    #tau = l ** 2 * (1 ­ model.p) / (2 * N * model.weight_decay)
    #predictive_variance += tau **­1
    model.eval()
    return predictive_variance.tolist()

## general evaluation
def dropout_eval(dataset, x_val, y_val, model, args):
    print("using bayesian evaluation ...")
    model.eval()
    y_pred = []
    y_truth = []
    uncertain_score_list = []
    represent_all = torch.FloatTensor()
    target_all = torch.LongTensor()
    val_iter = dataset.gen_minibatch(x_val, y_val, args.batch_size, args, shuffle=True)
    for batch in val_iter:
        feature, target = batch[0], batch[1]
        #feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit, represent = model(feature)
        score = uncertain_score(feature, model)
        uncertain_score_list += score
        #loss = F.cross_entropy(logit, target, size_average=False)

        #avg_loss += loss.data[0]
        y_pred_cur = (torch.max(logit, 1)[1].view(target.size()).data).tolist()
        y_truth_cur = target.data.tolist()

        y_pred += y_pred_cur
        y_truth += y_truth_cur
        # corrects += (torch.max(logit, 1)
        #              [1].view(target.size()).data == target.data).sum()

        represent_all = torch.cat([represent_all, represent.data.cpu()], 0)
        target_all = torch.cat([target_all, target.data.cpu()], 0)

        # print("\n=== logit ===")
        # print(logit)
        # print("=== prediction ===")
        # print(y_pred_cur)
        # print("=== truth ===")
        # print(y_truth_cur)

    # apply idk ratio to filter out the uncertain instances.
    # for i in range(len(y_pred)):
    #     print(uncertain_score_list[i], y_pred[i], y_truth[i],sep='\t')
    if args.use_idk:
        indices, L_sorted = zip(*sorted(enumerate(uncertain_score_list), key=itemgetter(1), reverse=False))
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



