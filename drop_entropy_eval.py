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
import scipy
import scipy.stats

def logit_score(logit, top_k):
    score_list = []
    for idx in range(len(logit)):
        logit_i = logit[idx].data.cpu().numpy().tolist()
        indices, L_sorted = zip(*sorted(enumerate(logit_i), key=itemgetter(1), reverse=True))
        score_i = (top_k * L_sorted[0] - sum(L_sorted[1:top_k + 1])) / top_k
        score_list.append(score_i)
    return score_list

def drop_freq(y_probs):
    y_probs = np.array(y_probs)
    n = y_probs.shape[0]
    drop_score_list = []
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        counts = np.bincount(logit_i)
        max_count = np.max(counts)
        drop_score_list.append(max_count / n)
    return drop_score_list

def drop_entropy(y_prbos, mask_num):
    y_probs = np.array(y_prbos)
    entropy_list = []
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        bin_count = np.bincount(logit_i)
        mask = sorted(range(len(bin_count)), key=lambda i: bin_count[i])[:-mask_num]
        bin_count[mask] = 0
        count_probs = [i / len(bin_count) for i in bin_count]

        entropy = scipy.stats.entropy(count_probs)
        entropy_list.append(entropy)
    return entropy_list

def uncertain_score(feature, model, drop_num=20, mask_num=5):

    # dropout
    model.train()
    logit_probs = []
    logit_score_list = []
    y_probs = []
    for _ in range(drop_num):
        logit, represent = model(feature)

        logit_var = np.var(logit.data.cpu().numpy(), axis=1).tolist()
        logit_s = logit_score(logit, top_k=3)
        y_pred = (torch.max(logit, 1)[1].view(feature.size()[0]).data).tolist()

        logit_probs += [logit_var]
        y_probs += [y_pred]
        logit_score_list += [logit_s]
    drop_score = drop_entropy(y_probs, mask_num)

    predictive_mean = np.mean(y_probs, axis=0)
    predictive_variance = np.var(y_probs, axis=0)
    #tau = l ** 2 * (1 ­ model.p) / (2 * N * model.weight_decay)
    #predictive_variance += tau **­1
    model.eval()

    # logit
    # logit, represent = model(feature)

    return drop_score
    #return predictive_mean.tolist()
    #return predictive_variance.tolist()

## general evaluation
def drop_entropy_eval(dataset, x_val, y_val, model, args):
    print("using drop entropy evaluation ...")
    model.eval()
    y_pred = []
    y_truth = []
    uncertain_score_list = []
    represent_all = torch.FloatTensor()
    target_all = torch.LongTensor()
    val_iter = dataset.gen_minibatch(x_val, y_val, args.batch_size, args, shuffle=True)
    repr_output = []
    for batch in val_iter:
        feature, target = batch[0], batch[1]
        #feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit, represent = model(feature)
        score = uncertain_score(feature, model, args.drop_num, args.drop_mask)
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
    if args.output_repr:
        repr_np = represent_all.numpy()
        for i, repr in enumerate(repr_np):
            repr_str = str(target_all[i])
            for dim in repr:
                repr_str += '\t' + "%.3f" % dim
            repr_output.append(repr_str)
        with open('./output_repr.txt', 'w') as f:
            for repr_str in repr_output:
                f.write(repr_str + '\n')


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

    if args.use_human_idk:
        indices, L_sorted = zip(*sorted(enumerate(uncertain_score_list), key=itemgetter(1), reverse=False))
        idk_list = np.arange(0, 0.45, 0.05)
        for idk_ratio in idk_list:
            # print("=== idk_ratio: ", idk_ratio, " ===")
            test_num = int(len(L_sorted) * (1 - idk_ratio))
            indices_cur = list(indices[:test_num])
            y_truth_cur = [y_truth[i] for i in indices_cur]
            y_pred_cur = [y_pred[i] for i in indices_cur]

            human_indices = list(indices[test_num:])
            y_human = [y_truth[i] for i in human_indices]
            y_truth_cur = y_truth_cur + y_human
            y_pred_cur = y_pred_cur + y_human

            f1_score = show_results(dataset, y_truth_cur, y_pred_cur, represent_all, target_all)

    else:
        f1_score = show_results(dataset, y_truth, y_pred, represent_all, target_all)

    return f1_score



