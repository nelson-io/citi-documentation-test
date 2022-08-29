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

    # import model

    model = joblib.load(input_filepaths[4])

    #predicts


    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    pred_labels_train = np.rint(train_preds)
    pred_labels_test = np.rint(test_preds)
    accuracy_train = sklearn.metrics.accuracy_score(y_train.target, pred_labels_train)
    accuracy_test = sklearn.metrics.accuracy_score(y_test.target, pred_labels_test)


    print(f"Model accuracy in train is {accuracy_train} and in test is {accuracy_test}.")

    # save model and predicts

    joblib.dump(test_preds,output_filepaths[0])
    joblib.dump(train_preds,output_filepaths[1]) 

if __name__ == '__main__':

    main(    input_filepaths =     ['/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/X_train.parquet',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/X_test.parquet',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/y_train.parquet',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/y_test.parquet',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/models/lgbm_model.pkl'],

    output_filepaths = [
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/models/test_preds.pkl',
            '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/models/train_preds.pkl'
        ])