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
from eval import eval

def multiclass_metric_loss(represent, target, margin=10, class_num=2, start_idx=1):
    target_list = target.data.tolist()
    dim = represent.data.shape[1]
    indices = []
    for class_idx in range(start_idx, class_num + start_idx):
        indice_i = [i for i, x in enumerate(target_list) if x == class_idx]
        indices.append(indice_i)

    loss_intra = Variable(torch.FloatTensor([0])).cuda()
    num_intra = 0
    loss_inter = Variable(torch.FloatTensor([0])).cuda()
    num_inter = 0
    for i in range(class_num):
        # intra class loss
        indice_i = indices[i]
        for intra_i in range(len(indice_i)):
            for intra_j in range(intra_i + 1, len(indice_i)):
                r_i = represent[indice_i[intra_i]]
                r_j = represent[indice_i[intra_j]]
                dist_ij = (r_i - r_j).norm(2)
                loss_intra += 1 / dim * (dist_ij * dist_ij)
                num_intra += 1

        # inter class loss
        for j in range(i + 1, class_num):
            indice_j = indices[j]
            for inter_i in indice_i:
                for inter_j in indice_j:
                    r_i = represent[inter_i]
                    r_j = represent[inter_j]
                    dist_ik = (r_i - r_j).norm(2)
                    tmp = margin - 1 / dim * (dist_ik * dist_ik)
                    loss_inter += torch.clamp(tmp, min=0)
                    num_inter += 1
    if num_intra > 0:
        loss_intra = loss_intra / num_intra
    if num_inter > 0:
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter

def metric_loss(represent, target, margin):

    target_list = target.data.tolist()
    dim = represent.data.shape[1]
    indices = [i for i, x in enumerate(target_list) if x == 1]
    other_indices = list(set(range(0, len(target_list))) - set(indices))

    # no label 1 instances
    if len(indices) == 0:
        return Variable(torch.FloatTensor([0])).cuda(), Variable(torch.FloatTensor([0])).cuda()

    loss_intra = Variable(torch.FloatTensor([0])).cuda()
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

    loss_inter = Variable(torch.FloatTensor([0])).cuda()
    num_inter = 0
    for i in indices:
        for k in other_indices:
            r_i = represent[i]
            r_k = represent[k]
            dist_ik = (r_i - r_k).norm(2)
            tmp = margin - 1 / dim * (dist_ik * dist_ik)
            loss_inter += torch.clamp(tmp, min=0)
            num_inter += 1
    if num_inter > 0:
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter

def train(dataset, x_train, y_train, x_val, y_val, model, args):
    if args.cuda:
        print("training model in cuda ...")
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    print(model)
    for epoch in range(1, args.epochs+1):
        train_iter = dataset.gen_minibatch(x_train, y_train, args.batch_size, args, shuffle=True)
        for batch in train_iter:
            feature, target = batch[0], batch[1]
            #feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit, represent = model(feature)
            loss_target = F.cross_entropy(logit, target)

            loss_metric = Variable(torch.FloatTensor([0])).cuda()
            if args.metric:
                class_num = dataset.get_class_num()
                if class_num == 2:
                    #loss_intra, loss_inter = metric_loss(represent, target, margin=args.metric_margin)
                    loss_intra, loss_inter = multiclass_metric_loss(represent, target, margin=args.metric_margin,
                                                                    class_num=class_num, start_idx=0)
                elif class_num > 2:
                    loss_intra, loss_inter = multiclass_metric_loss(represent, target, margin=args.metric_margin, class_num=class_num)
                # try:
                #
                # except:
                #     loss_intra, loss_inter = metric_loss(represent, target, margin=10)
                #     a = 0
                loss_metric = loss_intra + loss_inter

            loss = loss_target + args.metric_param * loss_metric

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / args.batch_size
                try:
                    if args.metric:
                        sys.stdout.write(
                            '\rEpoch[{}] Batch[{}] - loss: {:.4f}'
                            '({:.4f}/{:.4f}) acc: {:.4f}%({}/{})'.format(epoch,
                                                                         steps,
                                                                         loss.data[0],
                                                                         loss_target.data[0],
                                                                         loss_metric.data[0],
                                                                         accuracy,
                                                                         corrects,
                                                                         args.batch_size))
                    else:
                        sys.stdout.write(
                            '\rEpoch[{}] Batch[{}] - loss: {:.4f} '
                            'acc: {:.4f}%({}/{})'.format(epoch,
                                                         steps,
                                                         loss.data[0],
                                                         accuracy,
                                                         corrects,
                                                         args.batch_size))
                except:
                    print("Unexpected error:", sys.exc_info()[0])

            if steps % args.test_interval == 0:
                dev_acc = eval(dataset, x_val, y_val, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            # if steps % args.save_interval == 0:
            #     save(model, args.save_dir, 'snapshot', steps)
            model.train()


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)