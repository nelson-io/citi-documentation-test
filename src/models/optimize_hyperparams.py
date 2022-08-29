import pandas as pd
import pyarrow.parquet as pq
import optuna
import lightgbm as lgb
import sklearn.metrics
import numpy as np
from sklearn.model_selection import KFold
import joblib





# predict model
def main(input_filepaths, output_filepaths):
    #load data
    X_train, X_test, y_train, y_test = map(lambda x: pq.read_table(x).to_pandas(), input_filepaths)

    # set objective function


    def objective(trial):
        folds = 5
        seed = 1
        shuffle = True
        kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
        accuracy = []
        for train_idx, valid_idx in kf.split(X_train, y_train.target):
            train_data = X_train.iloc[train_idx,:], y_train.target[train_idx]
            valid_data = X_train.iloc[valid_idx,:], y_train.target[valid_idx]
            dtrain = lgb.Dataset(train_data[0], label=train_data[1])

            param = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }

            gbm = lgb.train(param, dtrain)
            preds = gbm.predict(valid_data[0])
            pred_labels = np.rint(preds)
            accuracy.append(sklearn.metrics.accuracy_score(valid_data[1], pred_labels))
            return np.mean(accuracy)


    study = optuna.create_study(sampler= optuna.samplers.TPESampler(n_startup_trials= 100),
    direction="maximize")

    study.optimize(objective, n_trials=300)
    joblib.dump(study, output_filepaths[0])

if __name__ == '__main__':



    main(input_filepaths =     ['/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/X_train.parquet',
        '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/X_test.parquet',
        '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/y_train.parquet',
        '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/y_test.parquet'],

        output_filepaths = ['/home/jovyan/work/github_projs/citiproj/citi-documentation-test/models/study.pkl'])