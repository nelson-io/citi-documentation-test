# data processment and feature engineering
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


def main(input_filepaths, output_filepaths):

    # read data, set missing values and remove irrelevant features
    train_df = (pq
        .read_table(source = input_filepaths[0])
        .to_pandas()
        .replace(" ?", np.NaN)
        .drop(['fnlwgt'],axis=1))

    test_df = (pq
        .read_table(source= input_filepaths[1])
        .to_pandas()
        .replace(" ?", np.NaN)
        .drop(['fnlwgt'],axis=1))

    # define X_train, X_test, y_train, y_test
    X_train = train_df.drop(['income'],axis=1)
    y_train = pd.DataFrame({"target":[1 if val == " >50K" else 0 for val in train_df.income]})

    X_test = test_df.drop(['income'],axis=1)
    y_test = pd.DataFrame({"target":[1 if val == " >50K" else 0 for val in test_df.income]})
    # set numerical an categorical columns

    numericalcols = list(train_df.select_dtypes(exclude='object').columns)
    categoricalcols = [x for x in train_df.columns if x not in numericalcols + ["income"]]

    # get_dummies

    X_train = pd.get_dummies(X_train)

    X_test =(pd.get_dummies(X_test)
        .reindex(columns = X_train.columns, fill_value=0)
    )


    # transform numerical features
    M = StandardScaler()
    X_train[numericalcols] = M.fit_transform(X_train[numericalcols])
    X_test[numericalcols] = M.transform(X_test[numericalcols])


    # write data in parquets

    X_train.to_parquet(output_filepaths[0])
    X_test.to_parquet(output_filepaths[1])
    y_train.to_parquet(output_filepaths[2])
    y_test.to_parquet(output_filepaths[3])

if __name__ == '__main__':



    main(input_filepaths=[
        "/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/interim/train_df.parquet",
        "/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/interim/test_df.parquet"],
        output_filepaths= [
        '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/X_train.parquet',
        '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/X_test.parquet',
        '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/y_train.parquet',
        '/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/processed/y_test.parquet'

      ])

