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



def metric_dist(represent, target):

    target_list = target.tolist()
    dim = represent.shape[1]
    indices = [i for i, x in enumerate(target_list) if x == 1]
    other_indices = list(set(range(0, len(target_list))) - set(indices))

    loss_intra = 0
    num_intra = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            r_i = represent[indices[i]]
            r_j = represent[indices[j]]
            dist_ij = (r_i - r_j).norm(2)
            loss_intra += 1 / dim * (dist_ij * dist_ij)
            num_intra += 1
    if num_intra > 0:
        loss_intra = loss_intra / num_intra

    loss_inter = 0
    num_inter = 0
    for i in indices:
        for k in other_indices:
            r_i = represent[i]
            r_k = represent[k]
            dist_ik = (r_i - r_k).norm(2)
            loss_inter += 1 / dim * (dist_ik * dist_ik)
            num_inter += 1
    if num_inter > 0:
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter


def open_dist(val_repr_list, val_target_list, test_repr_list, lambda_inter = 1):

    target_list = val_target_list.tolist()
    dim = val_repr_list.shape[1]
    indices = [i for i, x in enumerate(target_list) if x == 1]
    other_indices = list(set(range(0, len(target_list))) - set(indices))


    num_intra = len(indices)
    num_inter = len(other_indices)
    dist_list = []
    k = 3
    for idx in range(len(test_repr_list)):
        repr = test_repr_list[idx]
        loss_intra = 0
        loss_inter = 0

        cur_dist_list = []
        for i in indices:
            dist_intra = (val_repr_list[i] - repr).norm(2)
            cur_dist_list.append(1 / dim * dist_intra * dist_intra)

            #loss_intra += 1 / dim * dist_intra * dist_intra
        nearest_k = sorted(cur_dist_list)[0:k]
        dist = sum(nearest_k) / k

        # for i in other_indices:
        #     dist_inter = (val_repr_list[i] - test_repr).norm(2)
        #     loss_inter += 1 / dim * dist_inter * dist_inter

        #dist = loss_intra / num_intra
        dist_list.append(dist)

    return dist_list

def open_dist2(val_repr_list, val_target_list, test_repr_list, lambda_inter = 1):

    target_list = val_target_list.tolist()
    dim = val_repr_list.shape[1]
    indices = [i for i, x in enumerate(target_list) if x == 1]
    other_indices = list(set(range(0, len(target_list))) - set(indices))


    num_intra = len(indices)
    num_inter = len(other_indices)
    dist_list = []
    intra_list = []
    inter_list = []
    k = 10
    for idx in range(len(test_repr_list)):
        repr = test_repr_list[idx]
        loss_intra = 0
        loss_inter = 0

        cur_intra_dist = []
        for i in indices:
            dist_intra = (val_repr_list[i] - repr).norm(2)
            cur_intra_dist.append(1 / dim * dist_intra * dist_intra)

        intra_nearest_k = sorted(cur_intra_dist)[0:k]
        intra_dist = sum(intra_nearest_k) / k

        cur_inter_dist = []
        for i in other_indices:
            dist_inter = (val_repr_list[i] - repr).norm(2)
            cur_inter_dist.append(1 / dim * dist_inter * dist_inter)

        inter_nearest_k = sorted(cur_inter_dist)[0:k]
        inter_dist = sum(inter_nearest_k) / k

        #dist = loss_intra / num_intra
        intra_list.append(intra_dist)
        inter_list.append(inter_dist)
        dist_list.append(abs(inter_dist - intra_dist))

    return dist_list, intra_list, inter_list

def open_eval2(dataset, x_val, y_val, x_test, y_test, model, args):
    print("using open evaluation ...")
    model.eval()

    # collect the representation and target from test data set
    test_repr_list = torch.FloatTensor()
    test_target_list = torch.LongTensor()
    test_iter = dataset.gen_minibatch(x_test, y_test, args.batch_size, args, shuffle=True)
    y_pred = []
    for batch in test_iter:
        feature, target = batch[0], batch[1]
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit, represent = model(feature)

        test_repr_list = torch.cat([test_repr_list, represent.data.cpu()], 0)
        test_target_list = torch.cat([test_target_list, target.data.cpu()], 0)

        y_pred_cur = (torch.max(logit, 1)[1].view(target.size()).data).tolist()
        y_pred += y_pred_cur

    y_truth = test_target_list.tolist()

    # apply idk ratio to filter out the uncertain instances.
    if args.use_idk:
        val_repr_list = torch.FloatTensor()
        val_target_list = torch.LongTensor()
        val_iter = dataset.gen_minibatch(x_val, y_val, args.batch_size, args, shuffle=True)
        for batch in val_iter:
            feature, target = batch[0], batch[1]
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit, represent = model(feature)

            val_repr_list = torch.cat([val_repr_list, represent.data.cpu()], 0)
            val_target_list = torch.cat([val_target_list, target.data.cpu()], 0)

        dist_list, intra_list, inter_list = open_dist2(val_repr_list, val_target_list, test_repr_list)
        indices, L_sorted = zip(*sorted(enumerate(dist_list), key=itemgetter(1), reverse=True))

        for i in range(len(dist_list)):
            print(dist_list[i], intra_list[i], inter_list[i], y_pred[i], y_truth[i], sep='\t')

        idk_list = np.arange(0, 0.45, 0.05)
        for idk_ratio in idk_list:
            print("=== idk_ratio: ", idk_ratio, " ===")
            test_num = int(len(L_sorted) * (1 - idk_ratio))
            indices_cur = list(indices[:test_num])
            y_truth_cur = [y_truth[i] for i in indices_cur]
            y_pred_cur = [y_pred[i] for i in indices_cur]
            f1_score = show_results(y_truth_cur, y_pred_cur, test_repr_list, test_target_list)

    else:
        f1_score = show_results(y_truth, y_pred, test_repr_list, test_target_list)

    return f1_score


def open_eval(dataset, x_val, y_val, x_test, y_test, model, args):
    print("using open evaluation ...")
    model.eval()

    # collect the representation and target from test data set
    test_repr_list = torch.FloatTensor()
    test_target_list = torch.LongTensor()
    test_iter = dataset.gen_minibatch(x_test, y_test, args.batch_size, args, shuffle=True)
    y_pred = []
    for batch in test_iter:
        feature, target = batch[0], batch[1]
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit, represent = model(feature)

        test_repr_list = torch.cat([test_repr_list, represent.data.cpu()], 0)
        test_target_list = torch.cat([test_target_list, target.data.cpu()], 0)

        y_pred_cur = (torch.max(logit, 1)[1].view(target.size()).data).tolist()
        y_pred += y_pred_cur

    y_truth = test_target_list.tolist()

    # apply idk ratio to filter out the uncertain instances.
    if args.use_idk:
        # logit_diff_abs = [abs(x) for x in logit_diff]
        # indices, L_sorted = zip(*sorted(enumerate(logit_diff_abs), key=itemgetter(1), reverse=True))
        # collect the representation and target from eval data set
        val_repr_list = torch.FloatTensor()
        val_target_list = torch.LongTensor()
        val_iter = dataset.gen_minibatch(x_val, y_val, args.batch_size, args, shuffle=True)
        for batch in val_iter:
            feature, target = batch[0], batch[1]
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit, represent = model(feature)

            val_repr_list = torch.cat([val_repr_list, represent.data.cpu()], 0)
            val_target_list = torch.cat([val_target_list, target.data.cpu()], 0)

        dist_list = open_dist(val_repr_list, val_target_list, test_repr_list)
        pos_dist = []
        neg_dist = []
        pos_target = []
        neg_target = []
        pos_pred = []
        neg_pred = []
        for i in range(len(dist_list)):
            pred = y_pred[i]

            if pred == 1:
                pos_dist.append(dist_list[i])
                pos_target.append(y_truth[i])
                pos_pred.append(y_pred[i])
            else:
                neg_dist.append(dist_list[i])
                neg_target.append(y_truth[i])
                neg_pred.append(y_pred[i])

        pos_indices, pos_dist_sorted = zip(*sorted(enumerate(pos_dist), key=itemgetter(1), reverse=False))
        neg_indices, neg_dist_sorted = zip(*sorted(enumerate(neg_dist), key=itemgetter(1), reverse=True))

        idk_list = np.arange(0, 0.45, 0.05)
        for idk_ratio in idk_list:

            print("using idk_ratio", idk_ratio)

            pos_num_cur = int(len(pos_dist_sorted) * (1 - idk_ratio))
            pos_indices_cur = list(pos_indices[:pos_num_cur])
            pos_target_cur = [pos_target[i] for i in pos_indices_cur]
            pos_pred_cur = [pos_pred[i] for i in pos_indices_cur]


            neg_num_cur = int(len(neg_dist_sorted) * (1 - idk_ratio))
            #neg_num_cur = int(len(y_truth) * (1 - idk_ratio))
            neg_indices_cur = list(neg_indices[:neg_num_cur])
            neg_target_cur = [neg_target[i] for i in neg_indices_cur]
            neg_pred_cur = [neg_pred[i] for i in neg_indices_cur]

            y_truth_cur = pos_target_cur + neg_target_cur
            y_pred_cur = pos_pred_cur + neg_pred_cur

            # y_truth_cur = pos_target + neg_target_cur
            # y_pred_cur = pos_pred + neg_pred_cur

            f1_score = show_results(dataset, y_truth_cur, y_pred_cur, test_repr_list, test_target_list)

    else:
        f1_score = show_results(dataset, y_truth, y_pred, test_repr_list, test_target_list)

    return f1_score