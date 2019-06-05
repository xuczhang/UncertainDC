import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def load_weight_matrix(embed_dim, vocab_file):
    glove_dir = "./data/glove.6B/"
    glove_file = glove_dir + 'glove.6B.' + str(embed_dim) + 'd.txt'
    embeddings_index = {}
    f = open(glove_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
    f.close()

    with open(vocab_file) as f:
        vocab_words = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    vocab_words = [x.strip() for x in vocab_words]
    vocab_wordidx = {w: i for i, w in enumerate(vocab_words)}

    weights_matrix = np.random.random((len(vocab_wordidx), embed_dim))
    for word, i in vocab_wordidx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            weights_matrix[i] = embedding_vector
    return weights_matrix

class LSTM_Text(nn.Module):

    def __init__(self, args):
        super(LSTM_Text, self).__init__()
        self.args = args

        self.batch_size = args.batch_size
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.class_num = args.class_num
        self.hidden_size = 100

        self.model_type = args.model_type
        self.metric_fc_dim = 50

        self.vocab_file = args.data_path + args.dataset + '/vocabulary.txt'
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

        # Init with Glove Embedding
        if self.args.glove and not args.test:
            # load pre-trained glove embeddings
            print("Loading Glove embeddings ...")
            weights_matrix = load_weight_matrix(self.embed_dim, self.vocab_file)
            # weights_matrix = 0
            self.embed.weight.data.copy_(torch.from_numpy(weights_matrix))
            # self.embed.load_state_dict({'weight': weights_matrix})
            print("Glove embeddings loaded ...")

        # embed dropout
        if args.embed_dropout > 0:
            self.embed_dropout = nn.Dropout(args.embed_dropout)

        self.rnn = nn.LSTM(
            input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)

        # self.bn2 = nn.BatchNorm1d(self.hidden_size * 2)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.class_num)

        # just for trained model, remove the following two lines later
        # self.fc2 = nn.Linear(len(Ks) * Co, self.metric_fc_dim)
        # self.fc3 = nn.Linear(self.metric_fc_dim, C)

        # self.batchnorm = nn.BatchNorm1d()

    def forward(self, x):

        # N -- Batch Number
        # L -- Document Length
        # D -- Embed Dimension
        # d -- Hidden Dimension
        x = self.embed(x)  # (N, L, D)

        if self.args.static:
            x = Variable(x)

        x, _ = self.rnn(x)  # (N, L, 2d)
        x = x[:, -1, :]
        # x = self.bn2(x)
        x = self.dropout(x)  # (N, L, 2d)

        logit = self.fc1(x)  # (N, C)
        return logit, x