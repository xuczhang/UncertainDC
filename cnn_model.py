import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.class_num = args.class_num
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.in_channels = 1

        self.model_type = args.model_type
        self.metric_fc_dim = 50

        self.vocab_file = './data/' + args.dataset + '/vocabulary.txt'
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

        # Init with Glove Embedding
        if self.args.glove and not args.test:
            # load pre-trained glove embeddings
            print("Loading Glove embeddings ...")
            weights_matrix = load_weight_matrix(self.embed_dim, self.vocab_file)
            #weights_matrix = 0
            self.embed.weight.data.copy_(torch.from_numpy(weights_matrix))
            #self.embed.load_state_dict({'weight': weights_matrix})
            print("Glove embeddings loaded ...")

        # embed dropout
        if args.embed_dropout > 0:
            self.embed_dropout = nn.Dropout(args.embed_dropout)

        self.convs1 = nn.ModuleList([nn.Conv2d(self.in_channels, self.kernel_num, (K, self.embed_dim)) for K in self.kernel_sizes])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

        # just for trained model, remove the following two lines later
        # self.fc2 = nn.Linear(len(Ks) * Co, self.metric_fc_dim)
        # self.fc3 = nn.Linear(self.metric_fc_dim, C)

        #self.batchnorm = nn.BatchNorm1d()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        # add new dropout
        if self.args.embed_dropout > 0:
            x = self.embed_dropout(x)
        #x = self.dropout(x)
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''

        x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc1(x)  # (N, C)
        return logit, x