
# Code

The code for representation learning is based on the following work: "Unsupervised Scalable Representation Learning for Multivariate Time Series" (Jean-Yves Franceschi, Aymeric Dieuleveut and Martin Jaggi) [[NeurIPS]](https://papers.nips.cc/paper/8713-unsupervised-scalable-representation-learning-for-multivariate-time-series). The following requirements are mostly from their README file.

## Requirements

Experiments were done with the following package versions for Python 3.8 in Conda env:
conda install scikit-learn
conda install six
conda install psutil
conda install pandas
conda install numpy==1.21.4 matplotlib==3.4.3

This code should execute correctly with updated versions of these packages.

## Datasets

The datasets are collected from the SUMO scenario.

### Core
The `losses` and `networks` files are from the "Unsupervised Scalable Representation Learning for Multivariate Time Series".
 - `losses` folder: implements the triplet loss in the cases of a training set
   with all time series of the same length, and a training set with time series
   of unequal lengths;
 - `networks` folder: implements encoder and its building blocks (dilated
   convolutions, causal CNN);
 - `utils.py` file: common functions for three prediction methods.
 - `orig_Centralized.py` file: centralized training method.
 - `orig_Fed.py` file: general FL with FedAge aggregation.
 - `orig_WeightedLoss.py` file: the FL with weighted score method for DVFL.
 - `NUMO_load_data.py` file: functions for handling the data loading. 
 - `numo_data_collection_revised` file: the file that we use to execute SUMO with NUMO scenario and collect the features for traffic light prediction.
 - `hei_WeightedLoss.py` file: a random vehicle is selected as the aggregator within each community to perform local aggregation in the hierarchical structure for DVFL.
- `hie_WLoss_Alg.py` file: includes the Ng selection formula for choosing the optimal Ng to construct the hierarchical structure for DVFL.
- `hie_WLoss_Aggregator_Selection.py` file: the aggregator assignment algorithm is used to assign aggregators based on a max-heap built with randomly assigned CPU frequencies.

## Extra Files
 - `Attention*` files: All the files are further applied attention mechanism. These are not includes in the eventual result because the model become complex and takes much more time and rounds to train. However, it should be albe to reduce the MSE.

### Training on the UCR and UEA archives

For example, to train a causal dCNN mode with weighted loss:
`python ./WeightedLoss.py --log_filename ./Weighted.txt --processed_data_filename 'cent1' --num_total_clients 100 --learning_rate 0.003 --batch_size 16 --num_epoch 3 --isCommu True --isDCNN 1 --raw_data_filename 'cent1' --num_round 100`

Note:
1. There is no --num_round in Centralized
2. If no V2V communication, just remove `--isCommu True`. Do not put `--isCommu False`# DVFL
