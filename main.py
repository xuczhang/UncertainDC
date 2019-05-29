import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from GPUtil import GPUtil
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer

from datareader_newsgrp import load_newsgrp_data
from eval_dropent import dropent_eval_model
from utils import init_model
from eval import eval_model

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description='Uncertainty in Document Classification')

# training parameters
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate [default: 1e-4]')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for train [default: 50]')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training [default: 64]')
parser.add_argument('--seed', type=int, default=1, help='seed of random numbers [default: 64]')
parser.add_argument('--device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('--toy_data', action='store_true', default=False,
                    help='use an extremely small dataset for testing purpose [default: False]')
parser.add_argument('--val_ratio', type=float, default=0.3, help='validation rate [default: 0.3]')

parser.add_argument('--max_seq_len', type=int, default=300,
                    help='maximum sequence length which is necessary to limit memory usage [default: 200]')
parser.add_argument('--max_vocab_size', type=int, default=100000, help='maximum vocabulary size [default: 100000]')

# dataset
parser.add_argument('--dataset', type=str, default='newsgrp',
                    help='choose dataset to run [options: HIVE, COLLECTIONS]')
# model
parser.add_argument('--model', type=str, default="bert",
                    help='cnn, lstm, transformer, bert [default: bert]')
parser.add_argument('--bio', type=str, default="",
                    help='short bio to distinguish other models [default: ""]')
parser.add_argument('-num_filters', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-filters', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

# data
parser.add_argument('--data_path', type=str, default='./data/', help='the data directory')
# parser.add_argument('--encoder', type=str, default='bert', help='the type of encoder, bert or single token')
parser.add_argument('--snapshot_path', type=str, default="./snapshot/",
                    help='path of snapshot models [default: ./snapshot/]')

# testing
parser.add_argument('--test', action='store_true', default=False, help='use test mode')
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot [default: None]')

args = parser.parse_args()

# update args and print
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.model == "bert":
    args.encoder = "bert"
else:
    args.encoder = "single"

# initialize the snapshot path
now = datetime.now()  # current date and time
date_time = now.strftime("%Y%m%d_%H%M%S")
bio_str = "[" + args.bio + "]" if args.bio else ""
args.snapshot_path = args.snapshot_path + args.model + bio_str + "_" + args.dataset + "_" + date_time + "/"
if not args.toy_data and not args.test and not os.path.isdir(args.snapshot_path):
    os.makedirs(args.snapshot_path)

print("========== Parameter Settings ==========")
for arg in vars(args):
    print(arg, "=", getattr(args, arg), sep="")
print("========== ========== ==========")

# load data
DATA_ROOT = args.data_path + "newsgrp/"
reader, train_ds, test_ds, num_class = load_newsgrp_data(DATA_ROOT, args.encoder, args.max_seq_len,
                                         args.toy_data)

# choose available gpu
if args.device < 0:
    avail_gpus = GPUtil.getFirstAvailable(maxMemory=0.1, maxLoad=0.05)
    if len(avail_gpus) == 0:
        print("No GPU available!!!")
        sys.exit()
    args.device = avail_gpus[0]
    print("### Use GPU", str(args.device), "###")
torch.cuda.set_device(args.device)

# split the training set and validation set
val_size = int(len(train_ds) * args.val_ratio)
val_ds = train_ds[-val_size:]
train_ds = train_ds[:-val_size]

# prepare vocabulary
vocab = Vocabulary.from_instances(train_ds, max_vocab_size=args.max_vocab_size)

# prepare iterator
iterator = BucketIterator(batch_size=args.batch_size,
                          sorting_keys=[("tokens", "num_tokens")],
                          )
# tell the iterator how to numericalize the text data
iterator.index_with(vocab)

# read sample
# batch = next(iter(iterator(train_ds)))
if not args.test:
    print("Training Model ...")

    # initialize model
    model = init_model(args, args.model, vocab, num_class)

    # move model to gpu
    if args.cuda:
        model.cuda()

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    serialization_dir = args.snapshot_path if not args.toy_data else None
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        serialization_dir=serialization_dir,
        validation_dataset=val_ds,
        validation_metric="+accuracy",
        cuda_device=args.device if args.cuda else -1,
        num_epochs=args.epochs,
        num_serialized_models_to_keep=args.epochs,
        grad_norm=0.5,
        grad_clipping=1
    )
    metrics = trainer.train()

    # evaluate model after training
    dropent_eval_model(model, vocab, test_ds, args.batch_size, args.device if args.cuda else -1)
    # eval_model(model, vocab, test_ds, args.batch_size, args.device if args.cuda else -1)

elif args.test:  # test the single model
    # Evaluation
    print("Evaluation ...")

    if args.snapshot == None:
        print("No snapshot is provided!")
        exit(0)

    # load model from file
    model = init_model(args, args.model, vocab, num_class)

    with open(args.snapshot, 'rb') as f:
        model.load_state_dict(torch.load(f))
    if args.cuda:
        model.cuda(args.device)

    # run evaluation
    result = eval_model(model, vocab, test_ds, args.batch_size, args.device if args.cuda else -1)

# predicted = np.argmax(test_preds, axis=1)
# show_results(truth, predicted)
pass
