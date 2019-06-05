## Introduction
This is a Pytorch implementation of  neural-network-based uncertainty model for the task of text classification, as described in our NAACL 2019 paper:

**Mitigating Uncertainty in Document Classification**

## Requirement
* python 3
* pytorch > 0.1
* numpy

## Run the demo
```
python3 main.py 
```

## Data

We conducted experiments on three public datasets:
* 20 Newsgroups: a collection of 20,000 documents, partitioned evenly across 20 different newsgroups (20 classes)
* IMDb  Reviews: a collection of 50,000 popular movie reviews with binary positive or negative labels from the IMDb website (2 classes)
* Amazon  Reviews: a collection of reviews rating from 1 to 5 from Amazon between May 1996 and July 2013 (5 classes)

You can specify a dataset as follows:
```
python3 main.py -dataset=20news
```

## Usage
```
./main.py -h
```
or 

```
python3 main.py -h
```

You will get:

```
text classificer

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                initial learning rate [default: 0.001]
  -epochs EPOCHS        number of epochs for train [default: 100]
  -batch-size BATCH_SIZE
                        batch size for training [default: 32]
  -log-interval LOG_INTERVAL
                        how many steps to wait before logging training status
                        [default: 1]
  -test-interval TEST_INTERVAL
                        how many steps to wait before testing [default: 100]
  -save-interval SAVE_INTERVAL
                        how many steps to wait before saving [default:500]
  -save-dir SAVE_DIR    where to save the snapshot
  -early-stop EARLY_STOP
                        iteration numbers to stop without performance
                        increasing
  -save-best SAVE_BEST  whether to save when get best performance
  -data-path DATA_PATH  the data directory
  -dataset DATASET      choose dataset to run [options: 20news, imdb, amazon]
  -shuffle              shuffle the data every epoch
  -model MODEL          choose dataset to train [options: cnn, lstm]
  -model-type MODEL_TYPE
                        different structures of metric model, see document for
                        details
  -dropout DROPOUT      the probability for dropout [default: 0.3]
  -embed-dropout EMBED_DROPOUT
                        the probability for dropout [default: 0]
  -max-norm MAX_NORM    l2 constraint of parameters [default: 3.0]
  -embed-dim EMBED_DIM  number of embedding dimension [default: 200]
  -glove GLOVE          whether to use Glove pre-trained word embeddings
  -kernel-num KERNEL_NUM
                        number of each kind of kernel
  -kernel-sizes KERNEL_SIZES
                        comma-separated kernel size to use for convolution
  -static               fix the embedding
  -metric               use the metric learning
  -metric-param METRIC_PARAM
                        the parameter for the loss of metric learning
                        [default: 0.1]
  -metric-margin METRIC_MARGIN
                        the parameter for margin between different classes
                        [default: 0.1]
  -device DEVICE        device to use for iterate data, -1 mean cpu [default:
                        -1]
  -no-cuda              disable the gpu
  -snapshot SNAPSHOT    filename of model snapshot [default: None]
  -predict PREDICT      predict the sentence given
  -small SMALL          use the regular data or small data
  -test                 train or test
  -openev               use the open class for testing
  -dropev               use the dropout bayesian method for uncertainty
                        testing
  -dropentev            use the dropout bayesian method based on logit layer
                        for uncertainty testing
  -drop-mask DROP_MASK  the number of masks used for dropout bayesian method
                        [default: 5]
  -drop-num DROP_NUM    the number of the experiments used for dropout
                        bayesian method [default: 100]
  -distev               use the distance method for uncertainty testing
  -logitev              use the logit difference for uncertainty testing
  -logitev-topk LOGITEV_TOPK
                        the topk parameter for the loss of metric learning
                        [default: 5]
  -idk_ratio IDK_RATIO  the ratio of uncertainty
  -use_idk              use idk. If yes, it will show all the results from 0
                        to 0.4 with interval 0.05
  -use_human_idk        use human idk. If yes, it will show all the results
                        from 0 to 0.4 assuming the uncertain part is handed
                        over to humans
  -output_repr          output the representation to file output_repr.txt
```

## Train
You can specify a model to start training as follows:
```
python3 main.py -model=cnn
```
You can specify to use metric learning and set metric margin as follows:
```
python3 main.py -metric-margin=0.1
```
You will get:
```
Batch[100] - loss: 0.655424  acc: 59.3750%
```

## Test
After finish training the model, you can perform testing as follows:
```
/main.py -test -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps_1500.pt
```
The snapshot option means where to load your trained model. If you don't assign it, the model will start from scratch.

The proposed methods and baselines in the performance comparison can be performed as follows:

No Metric - PLV
```
python3 main.py -test -snapshot='./snapshot/stored/20news_no_metric.pt' -dataset=20news -use_idk
```
Metric - Distance
```
python3 main.py -test -snapshot='./snapshot/stored/20news_metric.pt' -dataset=20news -use_idk -distev
```
No Metric - Dropout
```
python3 main.py -test -snapshot='./snapshot/stored/20news_no_metric.pt' -dataset=20news -use_idk -dropev
```
Metric - Dropout
```
python3 main.py -test -snapshot='./snapshot/stored/20news_metric.pt' -dataset=20news -use_idk -dropev
```
No Metric - Dropout Entropy
```
python3 main.py -test -snapshot='./snapshot/stored/20news_no_metric.pt' -dataset=20news -use_idk -dropentev
```
Metric - Dropout Entropy
```
python3 main.py -test -snapshot='./snapshot/stored/20news_metric.pt' -dataset=20news -use_idk -dropentev
```

