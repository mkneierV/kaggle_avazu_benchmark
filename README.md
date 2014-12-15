kaggle_ctr_model
================

Kaggle Avazu beat-the-benchmark model. Very low memory implementation, scores 0.4037 with a lot of room for feature engineering improvements.


Usage:

1) Get the Data:
```
mkdir original_data
```
Download the data from https://www.kaggle.com/c/avazu-ctr-prediction/data into original_data directory.

2) Subset the data (feel free to use more or less):
```
mkdir submissions
mkdir modified_data
(head -1 original_data/train && tail -10000000 original_data/train) > modified_data/sub_train10
````
3) Run the model:
```
python run_model.py --neg_rate=.05 --submission_num=1 --n_iter=250 --train_path=modified_data/sub_train10
```

