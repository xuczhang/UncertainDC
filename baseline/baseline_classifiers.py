import argparse
import os
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators import BucketIterator

from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer

from br_data_reader import BRDataReader, load_br_data
from eval import Predictor, eval_ranking
from models.bert_classifier import BertClassifier
from models.bert_ctx_ranker import BertCtxRanker
from models.bert_noctx_ranker import BertNoCtxRanker
from models.lstm_classifier import LSTMClassifier

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description='Bug Triaging Model')

# training parameters
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate [default: 0.0001]')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for train [default: 50]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('--seed', type=int, default=1, help='seed of random numbers [default: 64]')
parser.add_argument('--device', type=int, default=2, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('--small_test', action='store_true', default=False, help='use a small dataset for testing purpose [default: False]')
parser.add_argument('--val_ratio', type=float, default=0.3, help='validation rate [default: 0.3]')

parser.add_argument('--max_seq_len', type=int, default=300, help='maximum sequence length which is necessary to limit memory usage [default: 200]')
parser.add_argument('--max_vocab_size', type=int, default=100000, help='maximum vocabulary size [default: 100000]')

parser.add_argument('--is_rank_task', type=bool, default=True, help='whether use the ranking task, otherwise it is classification task [default: True]')
# dataset
parser.add_argument('--dataset', type=str, default='HIVE', help='choose dataset to run [options: 20news, imdb, amazon, autogsr]')
# model
parser.add_argument('--model', type=str, default="bert_noctx", help='models: bert_ctx, bert_noctx, lstm_classifier, bert_classifier [default: bert_classifier]')
# data
parser.add_argument('--data-path', type=str, default='./data/', help='the data directory')

parser.add_argument('--snapshot-path', type=str, default="./snapshot/", help='path of snapshot models [default: ./snapshot/]')

# testing
parser.add_argument('--test', action='store_true', default=False, help='use test model')
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
#parser.add_argument('--test_ratio', type=float, default=0.2, help='test rate [default: 1]')
# parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
args = parser.parse_args()

# update args and print
args.cuda = torch.cuda.is_available()

PROJECT = "HIVE"
DATA_ROOT = Path("../dataset/bugtriaging/") / PROJECT
ASSIGNEE_ID_FILE = DATA_ROOT / "assignee_id.txt"
torch.manual_seed(args.seed)
torch.cuda.set_device(args.device)

# initialize the snapshot path
now = datetime.now()  # current date and time
date_time = now.strftime("%Y%m%d_%H%M%S")
args.snapshot_path = args.snapshot_path + args.model + "_" + PROJECT + "_" + date_time + "/"
if not args.test and not os.path.isdir(args.snapshot_path):
    os.makedirs(args.snapshot_path)

reader, train_ds, test_ds = load_br_data(DATA_ROOT, ASSIGNEE_ID_FILE, args.encoder, args.train_type, args.max_seq_len, args.toy_data)

# prepare vocabulary
vocab = Vocabulary.from_instances(train_ds, max_vocab_size=args.max_vocab_size)

# prepare iterator
iterator = BucketIterator(batch_size=args.batch_size,
                          sorting_keys=[("tokens", "num_tokens")],
                         )
# tell the iterator how to numericalize the text data
iterator.index_with(vocab)

# read sample
#batch = next(iter(iterator(train_ds)))
num_authors = reader.get_author_num()

if not args.test:
    print("Training Model ...")

    # initialize model
    if args.model == "bert_classifier":
        model = BertClassifier(
            args=args,
            out_sz=num_authors,
            vocab=vocab
        )
    elif args.model == "bert_ctx":
        model = BertCtxRanker(
            args=args,
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab
        )
    elif args.model == "bert_noctx":
        model = BertNoCtxRanker(
            args=args,
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab
        )
    elif args.model == "lstm_classifier":
        model = LSTMClassifier(
            args=args,
            out_sz=num_authors,
            vocab=vocab
        )

    # move model to gpu
    if args.cuda:
        model.cuda()

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        serialization_dir=args.snapshot_path,
        #validation_dataset=val_ds,
        cuda_device=args.device if args.cuda else -1,
        num_epochs=args.epochs,
    )
    metrics = trainer.train()

else:
    # Evaluation
    print("Evaluation ...")
    # iterate over the dataset without changing its order
    seq_iterator = BasicIterator(batch_size=args.batch_size)
    seq_iterator.index_with(vocab)

    if args.snapshot == None:
        print("No snapshot is provided!")
        exit(0)

    # load model from file
    if args.model == "bert_ctx":
        model = BertCtxRanker(
            args=args,
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab
        )
    elif args.model == "bert_noctx":
        model = BertNoCtxRanker(
            args=args,
            num_authors=num_authors,
            out_sz=num_authors,
            vocab=vocab
        )

    with open(args.snapshot, 'rb') as f:
        model.load_state_dict(torch.load(f))
    if args.cuda:
        model.cuda(args.device)

    predictor = Predictor(model, seq_iterator, cuda_device=args.device if args.cuda else -1)
    # train_preds = predictor.predict(train_ds)
    test_preds = predictor.predict(test_ds)
    truth = [i["label"].metadata for i in test_ds]

    for i in range(1, 11):
        ratio = i * 0.1
        test_size = int(len(test_ds) * ratio)
        result = eval_ranking(test_preds[:test_size], truth[:test_size])
        print("Result", str(ratio), ":", result)

    # result = eval_ranking(test_preds, truth)
    # print("Result:", result)


# predicted = np.argmax(test_preds, axis=1)
# show_results(truth, predicted)
pass