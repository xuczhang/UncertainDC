import os
from typing import *

import numpy as np
import pandas as pd
from allennlp.data import Instance, Field
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, LabelField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from overrides import overrides
from sklearn.datasets import fetch_20newsgroups


def download_newsgrp_data(data_root, train_df_file, test_df_file):
    """ Download the newsgrp data when it's never downloaded. """

    # load data
    categories = ['alt.atheism',
                  'comp.graphics',
                  'comp.os.ms-windows.misc',
                  'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware',
                  'comp.windows.x',
                  'misc.forsale',
                  'rec.autos',
                  'rec.motorcycles',
                  'rec.sport.baseball',
                  'rec.sport.hockey',
                  'sci.crypt',
                  'sci.electronics',
                  'sci.med',
                  'sci.space',
                  'soc.religion.christian',
                  'talk.politics.guns',
                  'talk.politics.mideast',
                  'talk.politics.misc',
                  'talk.religion.misc'
                  ]

    # train_categories = categories[0:15]
    train_categories = categories
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers'),
                                          categories=train_categories)

    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers'))

    # load vocab_words and vocab_wordidx
    # with open(self.vocab_words_file, 'rb') as f:
    #     (vocab_words, vocab_idx) = pickle.load(f)
    # self.vocab_size = len(vocab_words)

    train_df = pd.DataFrame(
        {'text': newsgroups_train.data,
         'label': newsgroups_train.target}
    )

    test_df = pd.DataFrame(
        {'text': newsgroups_test.data,
         'label': newsgroups_test.target}
    )

    train_df.to_pickle(data_root + train_df_file)
    test_df.to_pickle(data_root + test_df_file)

    return train_df, test_df


def load_newsgrp_data(data_root, encoder, max_seq_len, toy_data, is_rank_task=True):
    # the token indexer is responsible for mapping tokens to integers

    def bert_tokenizer(s: str):
        return token_indexer.wordpiece_tokenizer(s)[:max_seq_len - 2]

    def normal_tokenizer(x: str):
        return [w.text for w in
                SpacyWordSplitter(language='en_core_web_sm',
                                  pos_tags=False).split_words(x)[:max_seq_len]]

    if encoder == "bert":
        token_indexer = PretrainedBertIndexer(
            pretrained_model="bert-base-uncased",
            max_pieces=max_seq_len,
            do_lowercase=True,
        )
        tokenizer = bert_tokenizer
    else:
        token_indexer = SingleIdTokenIndexer()
        tokenizer = normal_tokenizer

    # init dataset reader
    reader = NewsGrpDataReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        toy_data=toy_data
    )

    train_pkl, test_pkl = "train.pkl", "test.pkl"

    # TODO: Check whether the pickle file exists
    if not os.path.isfile(data_root + train_pkl) or not os.path.isfile(data_root + test_pkl):
        download_newsgrp_data(data_root, train_pkl, test_pkl)

    train_ds, test_ds = (reader.read(data_root + fname) for fname in [train_pkl, test_pkl])
    num_class = reader.get_class_num()

    return reader, train_ds, test_ds, num_class


class NewsGrpDataReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = 300,
                 toy_data: bool = False) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len

        self.num_authors = -1
        self.toy_data = toy_data

        # from_date, last
        self.from_date, self.to_date = pd.Timestamp(2100, 1, 1).tz_localize('US/Eastern'), \
                                       pd.Timestamp(1900, 1, 1).tz_localize('US/Eastern')

    @overrides
    def text_to_instance(self, tokens: List[Token], label: int) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        id_field = MetadataField(id)
        fields["id"] = id_field

        assignee_field = LabelField(label, skip_indexing=True)
        fields["label"] = assignee_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:

        df = pd.read_pickle(file_path)

        # if config.testing:
        if self.toy_data:
            df = df.head(100)

        for i, row in df.iterrows():

            # text
            tokens = [Token(x) for x in self.tokenizer(row["text"])]

            # label
            label = row["label"]

            yield self.text_to_instance(
                tokens, label
            )

    def get_class_num(self):
        return 20

