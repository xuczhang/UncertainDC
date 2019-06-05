import glob
import json
import os
import scipy
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import operator
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from numpy.random import rand

class DataPlot:

    def __init__(self):
        self.init_plotting()
        pass

    def init_plotting(self):
        plt.rcParams['figure.figsize'] = (6.5, 5.5)
        plt.rcParams['font.size'] = 15
        #plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['legend.fontsize'] = 13
        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['savefig.dpi'] = plt.rcParams['savefig.dpi']
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['ytick.minor.size'] = 3
        plt.rcParams['ytick.major.width'] = 1
        plt.rcParams['ytick.minor.width'] = 1
        #plt.rcParams['legend.frameon'] = True
        #plt.rcParams['legend.loc'] = 'center left'
        #plt.rcParams['legend.loc'] = 'center left'
        plt.rcParams['axes.linewidth'] = 2

        #plt.gca().spines['right'].set_color('none')
        #plt.gca().spines['top'].set_color('none')
        #plt.gca().xaxis.set_ticks_position('bottom')
        #plt.gca().yaxis.set_ticks_position('left')

    def draw_imdb_accuracy(self, result_dir, category):

        result_file = result_dir + 'imdb_' + category.lower() + '.txt'
        acc_result = []
        with open(result_file, "r") as f:
            for line in f:
                acc_result.append([float(i) for i in line.split()])
        # mat_contents = scipy.io.loadmat(recovery_file)
        # Y_PLV = mat_contents["OLS_result"][0].tolist()
        # Y_Dist = mat_contents["RLHH_result"][0].tolist()
        # Y_Drop = mat_contents["OPAA_result"][0].tolist()
        # Y_DropMetric= mat_contents["ORL_result"][0].tolist()
        # Y_DE = mat_contents["ORL0_result"][0].tolist()
        # Y_DEMetric = mat_contents["BatchRC_result"][0].tolist()

        #x = [i*0.05 for i in range(2, 25)]
        x = [i*0.05 for i in range(0, len(acc_result[0]))]
        # plt.xticks(x, xticks)
        # begin subplots region
        # plt.subplot(121)
        plt.gca().margins(0.1, 0.1)
        ms = 7
        plt.plot(x, acc_result[0], linestyle='--', marker='d', markersize=ms, linewidth=3, color='#5461AA', label='PL-Variance')
        plt.plot(x, acc_result[1], linestyle='--', marker='o', markersize=ms, linewidth=3, color='green', label='Distance')
        plt.plot(x, acc_result[2], linestyle='-.', marker='v', markersize=ms, linewidth=3, color='blue', label='Dropout')
        plt.plot(x, acc_result[3], linestyle='-.', marker='<', markersize=ms, linewidth=3, color='#F27441', label='Dropout+Metric')
        plt.plot(x, acc_result[4], linestyle='--', marker='s', markersize=ms, linewidth=3, color='#BD90D4', label='DE')
        plt.plot(x, acc_result[5], linestyle=':', marker='^', markersize=ms, linewidth=3, color='cyan', label='DE+Metric')

        plt.xlabel(u'Uncertainty Elimination Ratio')
        plt.ylabel(category)

        # plt.xlim(1,len(Y_residual)+1)
        #plt.title(u'Subspace-Accuracy/NMI')

        # plt.yaxis.grid(color='gray', linestyle='dashed')

        #plt.gca().legend(bbox_to_anchor=(0.99, 0.99))
        #plt.gca().legend(bbox_to_anchor=(0.349, 1.005))
        #plt.gca().legend(loc = 'upper center', ncol=3)
        leg = plt.gca().legend(loc='upper left')
        leg.get_frame().set_alpha(0.7)
        #plt.yscale('log')
        #plt.ylim(0.96, 0.97)

        '''
        if b == 10 and k == 5 and p == 100 and bNoise == 1:
            plt.ylim(0.0, 0.5)
        elif b == 10 and k == 10 and p == 100 and bNoise == 1:
            plt.ylim(0.0, 0.4)
        elif b == 10 and k == 10 and p == 400 and bNoise == 1:
            plt.ylim(0.0, 0.5)  # used for 4K
        elif b == 10 and k == 1 and p == 200 and bNoise == 1:
            plt.ylim(0.15, 0.45)
        elif b == 10 and k == 2 and p == 200 and bNoise == 1:
            plt.ylim(0.1, 0.45)
        elif b == 10 and k == 2 and p == 200 and bNoise == 0:
            plt.ylim(-0.02, 0.45)
        elif b == 10 and k == 4 and p == 400 and bNoise == 0:
            plt.ylim(-0.05, 1.95)
        '''
        plt.show()

        #pp = PdfPages("D:/Dropbox/PHD/publications/IJCAI2017_RLHH/images/beta_1.pdf")
        #plt.savefig(pp, format='pdf')
        #plt.close()

    def draw_amazon_accuracy(self, result_dir, category):

        result_file = result_dir + 'amazon_' + category.lower() + '.txt'
        acc_result = []
        with open(result_file, "r") as f:
            for line in f:
                acc_result.append([float(i) for i in line.split()])
        # mat_contents = scipy.io.loadmat(recovery_file)
        # Y_PLV = mat_contents["OLS_result"][0].tolist()
        # Y_Dist = mat_contents["RLHH_result"][0].tolist()
        # Y_Drop = mat_contents["OPAA_result"][0].tolist()
        # Y_DropMetric= mat_contents["ORL_result"][0].tolist()
        # Y_DE = mat_contents["ORL0_result"][0].tolist()
        # Y_DEMetric = mat_contents["BatchRC_result"][0].tolist()

        #x = [i*0.05 for i in range(2, 25)]
        x = [i*0.05 for i in range(0, len(acc_result[0]))]
        # plt.xticks(x, xticks)
        # begin subplots region
        # plt.subplot(121)
        plt.gca().margins(0.1, 0.1)
        ms = 7
        plt.plot(x, acc_result[0], linestyle='--', marker='d', markersize=ms, linewidth=3, color='#5461AA', label='PL-Variance')
        plt.plot(x, acc_result[1], linestyle='--', marker='o', markersize=ms, linewidth=3, color='green', label='Distance')
        plt.plot(x, acc_result[2], linestyle='-.', marker='v', markersize=ms, linewidth=3, color='blue', label='Dropout')
        plt.plot(x, acc_result[3], linestyle='-.', marker='<', markersize=ms, linewidth=3, color='#F27441', label='Dropout+Metric')
        plt.plot(x, acc_result[4], linestyle='--', marker='s', markersize=ms, linewidth=3, color='#BD90D4', label='DE')
        plt.plot(x, acc_result[5], linestyle=':', marker='^', markersize=ms, linewidth=3, color='cyan', label='DE+Metric')

        plt.xlabel(u'Uncertainty Elimination Ratio')
        plt.ylabel(category)

        # plt.xlim(1,len(Y_residual)+1)
        #plt.title(u'Subspace-Accuracy/NMI')

        # plt.yaxis.grid(color='gray', linestyle='dashed')

        #plt.gca().legend(bbox_to_anchor=(0.99, 0.99))
        #plt.gca().legend(bbox_to_anchor=(0.349, 1.005))
        #plt.gca().legend(loc = 'upper center', ncol=3)
        leg = plt.gca().legend(loc='upper left')
        leg.get_frame().set_alpha(0.7)
        #plt.yscale('log')
        #plt.ylim(0.96, 0.97)

        '''
        if b == 10 and k == 5 and p == 100 and bNoise == 1:
            plt.ylim(0.0, 0.5)
        elif b == 10 and k == 10 and p == 100 and bNoise == 1:
            plt.ylim(0.0, 0.4)
        elif b == 10 and k == 10 and p == 400 and bNoise == 1:
            plt.ylim(0.0, 0.5)  # used for 4K
        elif b == 10 and k == 1 and p == 200 and bNoise == 1:
            plt.ylim(0.15, 0.45)
        elif b == 10 and k == 2 and p == 200 and bNoise == 1:
            plt.ylim(0.1, 0.45)
        elif b == 10 and k == 2 and p == 200 and bNoise == 0:
            plt.ylim(-0.02, 0.45)
        elif b == 10 and k == 4 and p == 400 and bNoise == 0:
            plt.ylim(-0.05, 1.95)
        '''
        plt.show()

        #pp = PdfPages("D:/Dropbox/PHD/publications/IJCAI2017_RLHH/images/beta_1.pdf")
        #plt.savefig(pp, format='pdf')
        #plt.close()

    def draw_metric_margin(self, result_dir):

        result_file = result_dir + 'metric_margin.txt'
        x = []
        macro_results = []
        micro_results = []

        with open(result_file, "r") as f:
            for line in f:
                results = line.split()
                x.append(float(results[0]))
                micro_results.append(float(results[1]))
                macro_results.append(float(results[2]))
        no_metric_micro = [0.849] * len(micro_results)
        no_metric_macro = [0.835] * len(micro_results)

        plt.gca().margins(0.1, 0.1)
        ms = 7.5
        # plt.plot(x, micro_results, linestyle='--', marker='o', markersize=ms, linewidth=3, color='#F27441',
        #          label='Micro-F1')
        # plt.plot(x, macro_results, linestyle='--', marker='d', markersize=ms, linewidth=3, color='#5461AA',
        #          label='Macro-F1')
        # plt.plot(x, no_metric_micro, linestyle='--', marker='s', markersize=0, linewidth=2, color='grey',
        #          label='No Metric Micro-F1')
        # plt.plot(x, no_metric_macro, linestyle='-.', marker='^', markersize=0, linewidth=2, color='#F27441',
        #          label='No Metric Macro-F1')

        plt.plot(x, micro_results, linestyle='-', marker='o', markersize=ms, linewidth=3, color='red',
                 label='Micro-F1')
        plt.plot(x, macro_results, linestyle='-', marker='s', markersize=ms, linewidth=3, color='k',
                 label='Macro-F1')
        plt.plot(x, no_metric_micro, linestyle='--', marker='d', markersize=0, linewidth=3, color='red',
                 label='No Metric Micro-F1')
        plt.plot(x, no_metric_macro, linestyle='-.', marker='^', markersize=0, linewidth=3, color='k',
                 label='No Metric Macro-F1')

        plt.xlabel(u'metric margin')
        plt.ylabel('F1 Scores')

        # plt.xlim(1,len(Y_residual)+1)
        #plt.title(u'Subspace-Accuracy/NMI')

        # plt.yaxis.grid(color='gray', linestyle='dashed')

        #plt.gca().legend(bbox_to_anchor=(0.99, 0.99))
        #plt.gca().legend(bbox_to_anchor=(0.349, 1.005))
        #plt.gca().legend(loc = 'upper center', ncol=3)
        leg = plt.gca().legend(loc='lower left')
        leg.get_frame().set_alpha(0.7)
        #plt.yscale('log')
        plt.xscale('log')
        #plt.ylim(0.96, 0.97)
        plt.xlim(0, 815)

        '''
        elif b == 10 and k == 4 and p == 400 and bNoise == 0:
            plt.ylim(-0.05, 1.95)
        '''
        plt.show()

        #pp = PdfPages("D:/Dropbox/PHD/publications/IJCAI2017_RLHH/images/beta_1.pdf")
        #plt.savefig(pp, format='pdf')
        #plt.close()

    def draw_embedding(self, result_dir):

        result_file = result_dir + 'embedding.txt'
        x = []
        ne_de = []
        ne_de_metric = []
        glove_de = []
        glove_de_metric = []
        with open(result_file, "r") as f:
            for line in f:
                results = line.split()
                x.append(float(results[0]))
                ne_de.append(float(results[1]))
                ne_de_metric.append(float(results[2]))
                glove_de.append(float(results[3]))
                glove_de_metric.append(float(results[4]))

        # mat_contents = scipy.io.loadmat(recovery_file)
        # Y_PLV = mat_contents["OLS_result"][0].tolist()
        # Y_Dist = mat_contents["RLHH_result"][0].tolist()
        # Y_Drop = mat_contents["OPAA_result"][0].tolist()
        # Y_DropMetric= mat_contents["ORL_result"][0].tolist()
        # Y_DE = mat_contents["ORL0_result"][0].tolist()
        # Y_DEMetric = mat_contents["BatchRC_result"][0].tolist()


        # plt.xticks(x, xticks)
        # begin subplots region
        # plt.subplot(121)
        plt.gca().margins(0.1, 0.1)
        ms = 8
        plt.plot(x, ne_de, linestyle='--', marker='d', markersize=ms, linewidth=3, color='#5461AA', label='No-Embed DE')
        plt.plot(x, ne_de_metric, linestyle='--', marker='o', markersize=ms, linewidth=3, color='#F27441', label='No-Embed DE+Metric')
        plt.plot(x, glove_de, linestyle='-.', marker='v', markersize=ms, linewidth=3, color='blue', label='Glove DE')
        plt.plot(x, glove_de_metric, linestyle='-.', marker='<', markersize=ms, linewidth=3, color='#F27441', label='Glove DE+Metric')
        # plt.plot(x, acc_result[4], linestyle='--', marker='s', markersize=ms, linewidth=3, color='#BD90D4', label='DE')
        # plt.plot(x, acc_result[5], linestyle=':', marker='^', markersize=ms, linewidth=3, color='cyan', label='DE+Metric')

        plt.xlabel(u'metric mdssmargin')
        plt.ylabel('F1 Score')

        # plt.xlim(1,len(Y_residual)+1)
        #plt.title(u'Subspace-Accuracy/NMI')

        # plt.yaxis.grid(color='gray', linestyle='dashed')

        #plt.gca().legend(bbox_to_anchor=(0.99, 0.99))
        #plt.gca().legend(bbox_to_anchor=(0.349, 1.005))
        #plt.gca().legend(loc = 'upper center', ncol=3)
        leg = plt.gca().legend(loc='upper right')
        leg.get_frame().set_alpha(0.7)
        #plt.yscale('log')
        #plt.xscale('log')
        #plt.ylim(0.96, 0.97)

        '''
        elif b == 10 and k == 4 and p == 400 and bNoise == 0:
            plt.ylim(-0.05, 1.95)
        '''
        plt.show()

        #pp = PdfPages("D:/Dropbox/PHD/publications/IJCAI2017_RLHH/images/beta_1.pdf")
        #plt.savefig(pp, format='pdf')
        #plt.close()

    def tnse_plot(self):

        # X = np.empty((0,200), float)
        # X_embedded = TSNE(n_components=2).fit_transform(X)
        # print(X_embedded.shape)
        color_plate = ['black', 'grey', 'rosybrown', 'red', 'tan',
                       'gold', 'olivedrab', 'chartreuse', 'darkgreen', 'deepskyblue',
                       'royalblue', 'navy', 'darkorchid', 'm', 'skyblue',
                       'slateblue', 'y', 'purple', 'tomato', 'gainsboro']

        # load represent data
        #repr_file = result_dir + 'output_repr_bak.txt'
        #repr_file = result_dir + 'repr_no_metric.txt'
        repr_file = result_dir + 'repr_metric0.5.txt'
        #repr_file = result_dir + 'repr_metric5.txt'
        #repr_file = result_dir + 'repr_metric10.txt'


        X = np.empty((0, 300), float)
        colors = []
        with open(repr_file, "r") as f:
            for line in f:
                results = line.split()
                target = int(results[0])
                repr = [float(i) for i in results[1:]]

                colors.append(target)
                X = np.append(X, np.array([repr]), axis=0)

        X_embedded = TSNE(n_components=2).fit_transform(X)

        fig, ax = plt.subplots()
        for i, repr in enumerate(X_embedded):
            x = repr[0]
            y = repr[1]
            scale = 6.0
            color = color_plate[colors[i]]
            #color = 'red'
            ax.scatter(x, y, c=color, s=scale,
                       alpha=0.8, edgecolors='none')

        #ax.legend()
        ax.grid(True)
        print("drawing ...")
        plt.show()

if __name__ == '__main__':

    data_plot = DataPlot()

    ''' beta recovery '''
    result_dir = '/Users/xuczhang/Dropbox/PHD/publications/EMNLP2018_Uncertainty/result/'
    # data_plot.draw_imdb_accuracy(result_dir, 'Accuracy')
    # data_plot.draw_imdb_accuracy(result_dir, 'F1')

    #data_plot.draw_metric_margin(result_dir)
    #data_plot.draw_embedding(result_dir)

    #data_plot.tnse_plot()

    data_plot.draw_amazon_accuracy(result_dir, 'Accuracy')
