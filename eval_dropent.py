from operator import itemgetter
from typing import Iterable

import numpy as np
import scipy
import scipy.stats
import torch
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator, BasicIterator
from allennlp.models import Model
from allennlp.nn import util as nn_util
from scipy.special import expit  # the sigmoid function
from tqdm import tqdm

from eval import show_results, tonp


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

def drop_entropy(probs, mask_num):
    probs = np.array(probs)
    entropy_list = []
    for i in range(probs.shape[1]):
        logit_i = probs[:, i]
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
    model.eval()

    return drop_score


def dropent_eval_model(model, vocab, test_ds, batch_size, device, verbose=True, use_idk=False, use_human_idk=True):
    # iterate over the dataset without changing its order
    seq_iterator = BasicIterator(batch_size=batch_size)
    seq_iterator.index_with(vocab)

    predictor = DropEntropyPredictor(model, seq_iterator, cuda_device=device)
    # train_preds = predictor.predict(train_ds)
    test_preds, uncertain_scores = predictor.predict(test_ds)
    truth = [i["label"].label for i in test_ds]

    result = show_results(truth, test_preds)
    result_str = "\t".join(['%.3f' % i for i in result]) + "\n"


    if use_idk:
        indices, L_sorted = zip(*sorted(enumerate(uncertain_scores), key=itemgetter(1), reverse=False))
        idk_list = np.arange(0, 0.45, 0.05)
        for idk_ratio in idk_list:
            #print("=== idk_ratio: ", idk_ratio, " ===")
            test_num = int(len(L_sorted) * (1 - idk_ratio))
            indices_cur = list(indices[:test_num])
            y_truth_cur = [truth[i] for i in indices_cur]
            y_pred_cur = [test_preds[i] for i in indices_cur]
            result = show_results(y_truth_cur, y_pred_cur)
            print("\t".join(['%.3f' % i for i in result]) + "\n")

    if use_human_idk:
        indices, L_sorted = zip(*sorted(enumerate(uncertain_scores), key=itemgetter(1), reverse=False))
        idk_list = np.arange(0, 0.45, 0.05)
        for idk_ratio in idk_list:
            # print("=== idk_ratio: ", idk_ratio, " ===")
            test_num = int(len(L_sorted) * (1 - idk_ratio))
            indices_cur = list(indices[:test_num])
            y_truth_cur = [truth[i] for i in indices_cur]
            y_pred_cur = [test_preds[i] for i in indices_cur]

            human_indices = list(indices[test_num:])
            y_human = [truth[i] for i in human_indices]
            y_truth_cur = y_truth_cur + y_human
            y_pred_cur = y_pred_cur + y_human

            result = show_results(y_truth_cur, y_pred_cur)
            print("\t".join(['%.3f' % i for i in result]) + "\n")

    if verbose:
        print(result_str)

    return result_str


class DropEntropyPredictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        self.drop_num = 50
        self.mask_num = 5

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["class_logits"]))

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))

        preds, uncertain_scores = [], []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)

                # prediction result
                self.model.eval()
                batch_logits = self._extract_data(batch)
                batch_preds = np.argmax(batch_logits, axis=1)
                preds.append(batch_preds)

                # uncertainty evaluation
                self.model.train()  # set to training mode for dropout uncertainty evaluation
                # logit_probs, logit_score_list, y_probs = [], [], []
                y_probs = []
                for _ in range(self.drop_num):
                    uc_logit = self._extract_data(batch)

                    # logit_var = np.var(uc_logit.data.cpu().numpy(), axis=1).tolist()
                    # logit_s = logit_score(uc_logit, top_k=3)
                    drop_preds = np.argmax(uc_logit, axis=1).tolist()

                    # logit_probs += [logit_var]
                    y_probs += [drop_preds]
                    # logit_score_list += [logit_s]
                drop_score = drop_entropy(y_probs, self.mask_num)
                uncertain_scores += drop_score

        preds = np.concatenate(preds, axis=0)

        return preds, uncertain_scores