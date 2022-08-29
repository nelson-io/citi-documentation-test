# make model

import optuna
import numpy as np
import pandas as pd 
import joblib
import pyarrow.parquet as pq
import sklearn.metrics
import lightgbm as lgb


def main(input_filepaths, output_filepaths):






    # predict model

    #load data
    X_train, X_test, y_train, y_test = map(lambda x: pq.read_table(x).to_pandas(), input_filepaths[:4])

    # import params

    study = joblib.load(input_filepaths[4])
    best_params = study.best_params

    #train model

    dtrain = lgb.Dataset(X_train, label=y_train.target)
    gbm = lgb.train(best_params, dtrain)


    # save model 

    joblib.dump(gbm, output_filepaths[0] )


if __name__ == '__main__':

    main(    input_filepaths =     ['/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/X_train.parquet',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/X_test.parquet',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/y_train.parquet',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/y_test.parquet',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/models/study.pkl'],

    output_filepaths = ['/home/jovyan/work/github_projs/citiproj/citi-documentation-test/models/lgbm_model.pkl'
        ])