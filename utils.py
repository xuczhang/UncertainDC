import os
import torch
from torch.autograd import Variable

from models.bert_classifier import BertClassifier
# from models.lstm_classifier import LSTMClassifier
from models.cnn_classifier import CNNClassifier
from models.lstm_classifier import LSTMClassifier
from models.transformer_classifier import TransformerClassifier


def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def save_snapshot(snap_path, save_prefix, model, vocab):

    if not os.path.isdir(snap_path):
        os.makedirs(snap_path)
    save_prefix = os.path.join(snap_path, save_prefix)
    save_path = '{}.pt'.format(save_prefix)

    print("Saving model to", save_path)
    with open(save_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(save_prefix + "vocab")

#save_snapshot(args.snapshot_path, "bert_rank", model, vocab)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

# a utility function to initialize the models
def init_model(args, model_type, vocab, num_class):

    model = None
    if model_type == "cnn":
        model = CNNClassifier(
            args=args,
            out_sz=num_class,
            vocab=vocab
        )
    elif model_type == "lstm":
        model = LSTMClassifier(
            args=args,
            out_sz=num_class,
            vocab=vocab
        )
    elif model_type == "transformer":
        model = TransformerClassifier(
            args=args,
            out_sz=num_class,
            vocab=vocab
        )
    elif model_type == "bert":
        model = BertClassifier(
            args=args,
            out_sz=num_class,
            vocab=vocab
        )

    return model


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))