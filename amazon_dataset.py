import nltk
from sklearn.datasets import fetch_20newsgroups

import platform
import re
import os
import random
import tarfile
import urllib
import pickle
import numpy as np
import itertools

import torch
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import os.path
import glob
import ntpath
import json

class Amazon_DataSet:

    def __init__(self, args):
        """Create an IMDB dataset instance. """

        #self.word_embed_file = self.data_folder + 'embedding/wiki.ar.vec'
        # word_embed_file = data_folder + "embedding/Wiki-CBOW"
        self.data_dir = args.data_path + 'amazon/'
        self.raw_data_dir = '/Users/xuczhang/Dataset/amazon/amazon_json/'
        self.vocab_file = self.data_dir + 'vocabulary.txt'
        self.train_df_file = self.data_dir + 'train_df.pkl'
        self.test_df_file = self.data_dir + 'test_df.pkl'
        self.lemmatizer = WordNetLemmatizer()
        self.class_num = -1
        pass

    def clean_str(self, string):
        """
        Tokenization/string cleaning.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab_file(self):
        return self.vocab_file

    def get_class_num(self):
        return self.class_num

    def cal_class_num(self, df):
        return 5
        # ids_labels = set()
        #
        # # since sometimes the data will be shuffled in the frame
        # # during train test split
        # for index in df.index:
        #     # labels
        #     ids_labels.add(df.Class[index])
        # return len(ids_labels)

    def load_vocab(self):

        # if not os.path.isfile(self.vocab_size):
        #     # generate the vocab file
        #     newsgroups = fetch_20newsgroups(remove=('headers'))
        #
        #     pass

        with open(self.vocab_file) as f:
            vocab_words = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        vocab_words = [x.strip() for x in vocab_words]

        self.vocab_size = len(vocab_words)
        vocab_wordidx = {w: i for i, w in enumerate(vocab_words)}

        return vocab_wordidx

    def sent_parse(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences_tokens = []

        for sentence in sentences:
            tokens = nltk.tokenize.wordpunct_tokenize(sentence)
            processed_tokens = []
            for token in tokens:
                token = self.clean_str(token)
                token = self.lemmatizer.lemmatize(token)
                if not token:
                    processed_tokens.append(token)
            sentences_tokens.append(tokens)
        return sentences_tokens

    def generate_data(self, val_ratio=.1, shuffle=True):
        print("generating amazon data ...")
        train_df, test_df, vocab_wordidx = self.load_data_if_not_exist()

        self.class_num = self.cal_class_num(train_df)

        train_df, val_df = train_test_split(train_df,
                                            test_size=val_ratio, random_state=967898,
                                            stratify=train_df.Class)

        x_train, y_train = self.format_data(train_df, vocab_wordidx)
        x_val, y_val = self.format_data(val_df, vocab_wordidx)
        x_test, y_test = self.format_data(test_df, vocab_wordidx)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


    def load_data(self, data_folder):


        text_list = []
        target_list = []
        for file in glob.glob(data_folder + "*.json"):

            with open(file, "r") as f:
                for line in f:
                    record = json.loads(line)
                    sent = self.sent_parse(record['reviewText'])
                    target = int(record['overall']) - 1
                    #print(sent)
                    text_list.append(sent)
                    target_list.append(target)
        return text_list, target_list

    def load_data_if_not_exist(self):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            args: arguments
            val_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
        """

        # check the existence of data files
        if os.path.isfile(self.train_df_file) and os.path.isfile(self.test_df_file):
            train_df = pd.read_pickle(self.train_df_file)
            test_df = pd.read_pickle(self.test_df_file)
            vocab_wordidx = self.load_vocab()
            return train_df, test_df, vocab_wordidx

        # load data


        # load vocab_words and vocab_wordidx
        # with open(self.vocab_words_file, 'rb') as f:
        #     (vocab_words, vocab_idx) = pickle.load(f)
        # self.vocab_size = len(vocab_words)

        data_text, data_target = self.load_data(self.raw_data_dir)

        df = pd.DataFrame(
            {'Text': data_text,
             'Class': data_target}
        )

        test_ratio = 0.2
        train_df, test_df = train_test_split(df,
                                            test_size=test_ratio, random_state=967898,
                                            stratify=df.Class)

        train_df.to_pickle(self.train_df_file)
        test_df.to_pickle(self.test_df_file)

        # extract title and sentence words
        train_sent_words = train_df.Text.apply(lambda ll: list(itertools.chain.from_iterable(ll)))
        train_sent_words = list(itertools.chain.from_iterable(train_sent_words))
        test_sent_words = test_df.Text.apply(lambda ll: list(itertools.chain.from_iterable(ll)))
        test_sent_words = list(itertools.chain.from_iterable(test_sent_words))

        # generate vocabulary words
        vocab_words = list(set(train_sent_words) | set(test_sent_words))
        # add extra words such as start/end of sentence
        vocab_words.append("<UNK>")
        vocab_words.append("<SOSent>")
        vocab_words.append("<EOSent>")
        vocab_words.append("<SODoc>")
        vocab_words.append("<EODoc>")

        vocab_words.sort()

        # save vocabulary words and word index into files
        with open(self.vocab_file, 'w') as f:
            for word in vocab_words:
                f.write(word)
                f.write('\n')
            #pickle.dump(vocab_words, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.vocab_size = len(vocab_words)
        vocab_wordidx = {w: i for i, w in enumerate(vocab_words)}

        return train_df, test_df, vocab_wordidx

    def format_data(self, data_frame, vocab_idx):

        ids_document = []
        ids_labels = []

        # since sometimes the data will be shuffled in the frame
        # during train test split
        for index in data_frame.index:
            document = data_frame.Text[index]
            text_word_list = [self.convertSent2WordIds(sentence, vocab_idx) for sentence in
                              document]
            #text_word_list = [j for j in i for i in text_word_list]
            text_word_list = [item for sublist in text_word_list for item in sublist]
            ids_document.append(text_word_list)

            # labels
            ids_labels.append(data_frame.Class[index])

        return np.array(ids_document), np.array(ids_labels)

    def convertSent2WordIds(self, sentence, vocab_idx):
        """
        sentence is a list of word.
        It is converted to list of ids based on vocab_idx
        """
        sentence_start_tag_idx = vocab_idx["<SOSent>"]
        sentence_end_tag_idx = vocab_idx["<EOSent>"]
        word_unknown_tag_idx = vocab_idx["<UNK>"]

        sent2id = [sentence_start_tag_idx]
        #sent2id = []

        try:
            sent2id = sent2id + [
                vocab_idx[word] if vocab_idx[word] < self.vocab_size else word_unknown_tag_idx for
                word in sentence]
        except KeyError as e:
            print(e)
            print(sentence)
            raise ValueError('Fix this issue dude')

        sent2id = sent2id + [sentence_end_tag_idx]
        return sent2id

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

        # return last part
        start_idx = inputs.shape[0] - int(inputs.shape[0]) % batchsize
        if shuffle:
            excerpt = indices[start_idx:]
        else:
            excerpt = slice(start_idx, inputs.shape[0])
        yield inputs[excerpt], targets[excerpt]

    def gen_minibatch(self, tokens, labels, mini_batch_size, args, shuffle=True):
        for token, label in self.iterate_minibatches(tokens, labels, mini_batch_size, shuffle=shuffle):
            token = self.pad_batch(token)
            token.data.t_()
            label = Variable(torch.from_numpy(label), requires_grad=False)
            if args.cuda == True:
                token, label = token.cuda(), label.cuda()

            yield (token, label)

    def pad_batch(self, mini_batch):
        mini_batch_size = len(mini_batch)
        #mean_sent_len = int(np.mean([len(x) for x in mini_batch]))
        mean_token_len = int(np.mean([len(x) for x in mini_batch]))
        max_token_len = int(np.max([len(x) for x in mini_batch]))
        main_matrix = np.zeros((mini_batch_size, mean_token_len), dtype=np.int)
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                    try:
                        main_matrix[i, j] = mini_batch[i][j]
                    except IndexError:
                        pass
        return Variable(torch.from_numpy(main_matrix).transpose(0, 1))


