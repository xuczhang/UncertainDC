from typing import Iterable

import sklearn
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator, BasicIterator
from allennlp.models import Model
from tqdm import tqdm
from scipy.special import expit  # the sigmoid function
import torch
import numpy as np
from allennlp.nn import util as nn_util
from utils import init_model, split

def tonp(tsr): return tsr.detach().cpu().numpy()


# def show_results(truth, pred):
#
#     accuracy_score = (sklearn.metrics.accuracy_score(truth, pred))
#     micro_f1_score = (sklearn.metrics.f1_score(truth, pred, average='micro'))
#     macro_f1_score = (sklearn.metrics.f1_score(truth, pred, average='macro'))
#     return accuracy_score, micro_f1_score, macro_f1_score


def show_results(truth_results, preds):

    class_num = len(set(truth_results))

    if class_num == 2:
        accuracy_score = (sklearn.metrics.accuracy_score(truth_results, preds))
        f1_score = (sklearn.metrics.f1_score(truth_results, preds, pos_label=1))
        prec_score = (sklearn.metrics.precision_score(truth_results, preds, pos_label=1))
        recall_score = (sklearn.metrics.recall_score(truth_results, preds, pos_label=1))
        confusion_mat = (sklearn.metrics.confusion_matrix(truth_results, preds))

        return accuracy_score, f1_score, prec_score, recall_score

    elif class_num > 2:
        accuracy_score = (sklearn.metrics.accuracy_score(truth_results, preds))
        micro_f1_score = (sklearn.metrics.f1_score(truth_results, preds, average='micro'))
        macro_f1_score = (sklearn.metrics.f1_score(truth_results, preds, average='macro'))
        return accuracy_score, micro_f1_score, macro_f1_score


def eval_model(model, vocab, test_ds, batch_size, device, verbose=True):
    # iterate over the dataset without changing its order
    seq_iterator = BasicIterator(batch_size=batch_size)
    seq_iterator.index_with(vocab)

    predictor = Predictor(model, seq_iterator, cuda_device=device)
    # train_preds = predictor.predict(train_ds)
    test_preds = predictor.predict(test_ds)
    truth = [i["label"].label for i in test_ds]

    result = show_results(truth, test_preds)
    result_str = "\t".join(['%.3f' % i for i in result]) + "\n"

    if verbose:
        print(result_str)

    return result_str


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["class_logits"]))

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                batch_logits = self._extract_data(batch)
                batch_preds = np.argmax(batch_logits, axis=1)
                #pred = (torch.max(logits, 1)[1].view(target.size()).data).tolist()
                preds.append(batch_preds)
        return np.concatenate(preds, axis=0)

