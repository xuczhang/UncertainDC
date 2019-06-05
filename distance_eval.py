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
import math
from eval import show_results


def confidence_score(val_repr_list, val_target_list, test_repr_list, test_target_list, top_k=10):

    correct_list = []
    total_list = []
    score_list = []
    for idx in range(len(test_repr_list)):
        repr = test_repr_list[idx]
        target = test_target_list[idx]
        for i in range(len(val_target_list)):
            val_target = val_target_list[i]
            val_repr = val_repr_list[i]
            dist = (val_repr - repr).norm(2)
            e_dist = math.exp(-dist)
            if val_target == target:
                correct_list.append(e_dist)
            total_list.append(e_dist)

        correct_k = sorted(correct_list)[0:top_k]
        total_k = sorted(total_list)[0:top_k]
        score_list.append(sum(correct_k) / sum(total_k))

    return score_list

## distance evaluation
def distance_eval(dataset, x_train, y_train, x_val, y_val, model, args):
    print("using distance evaluation ...")
    MAX_TRAIN_NUM = 200
    if len(y_train) > MAX_TRAIN_NUM and args.dataset == 'amazon':
        x_train = x_train[0:MAX_TRAIN_NUM]
        y_train = y_train[0:MAX_TRAIN_NUM]

    model.eval()
    y_pred = []
    y_truth = []

    # calculate the representation of training data
    train_repr_list = torch.FloatTensor()
    train_target_list = torch.LongTensor()
    train_iter = dataset.gen_minibatch(x_train, y_train, args.batch_size, args, shuffle=True)
    for batch in train_iter:
        feature, target = batch[0], batch[1]
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit, represent = model(feature)

        train_repr_list = torch.cat([train_repr_list, represent.data.cpu()], 0)
        train_target_list = torch.cat([train_target_list, target.data.cpu()], 0)
    
    
    confidence_score_list = []
    represent_all = torch.FloatTensor()
    target_all = torch.LongTensor()
    val_iter = dataset.gen_minibatch(x_val, y_val, args.batch_size, args, shuffle=True)
    for batch in val_iter:
        feature, target = batch[0], batch[1]
        #feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit, represent = model(feature)
        score = confidence_score(train_repr_list, train_target_list, represent.data.cpu(), target.data.cpu())
        confidence_score_list += score

        y_pred_cur = (torch.max(logit, 1)[1].view(target.size()).data).tolist()
        y_truth_cur = target.data.tolist()

        y_pred += y_pred_cur
        y_truth += y_truth_cur

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

        indices, L_sorted = zip(*sorted(enumerate(confidence_score_list), key=itemgetter(1), reverse=True))
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



