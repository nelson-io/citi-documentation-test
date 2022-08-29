# -*- coding: utf-8 -*-
# import click
import logging
import pandas as pd



def main(input_filepaths, output_filepaths):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #set names
    df_names = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]

    # read files and make train and test dataframes
    train_df = pd.read_csv(input_filepaths[0],
        names = df_names)

    test_df = pd.read_csv(input_filepaths[1],
        names = df_names, skiprows = 1)

    # write files into parquets

    train_df.to_parquet(output_filepaths[0])
    test_df.to_parquet(output_filepaths[1])


 


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main(input_filepaths=[
            "/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/raw/adult.data",
            "/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/raw/adult.test"],
        output_filepaths= [
            "/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/interim/train_df.parquet",
            "/home/jovyan/work/github_projs/citiproj/citi-documentation-test/data/interim/test_df.parquet"
      ])
