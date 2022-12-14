# Model Analysis Report

## Summary
This model attempts to solve the classification problem of detecting individuals with annual incomes above 50K USD from data obtained from the 1994 US census. 

## Data pre-processing
Data was available in csv format without headers. Headers were added and data was stored into a train set and a test set, both in parquet format [here](https://github.com/nelson-io/citi-documentation-test/tree/main/data/interim).

The data had missing values that were expressed with a string ("?") and were replaced into python numpy's class that can be handled by the machine learning framework used.

The feature that expressed the weight of every combination into the census was removed for methodological reasons since it didn't represent information from the original census and it was an estimate added afterwards.

Numerical data was scaled and categorical data one-hot-encoded in order to be able to use any kind of Machine Learning framework.

All the transformations were fitted with the train set prior to be applied to the test set in order to avoid data leakage.

The clean data, ready to be used in any model was stored in [data/processed](https://github.com/nelson-io/citi-documentation-test/tree/main/data/processed).


## Model selection

For practical reasons, instead of trying multiple models and making them compete, LightGBM was selected since it's usually performant and efficient.

Model was optimized with Optuna and the optimization results can be accessed in [reports/model_optimization](https://github.com/nelson-io/citi-documentation-test/blob/main/reports/model_optimization.md).

## Metrics

The model params were optimized in order to maximize the accuracy and after testing the results on the train set we obtained the following results:
* **Train Accuracy** : 0.880
* **Test Accuracy** :  0.875

Which suggests no overfitting. Since the target was unbalanced, it might make sense to carry on the optimization using an objective function with a metric more resilient to unbalanced designs or treating the data beforehand with under-sampling or over-sampling techniques.

The confusion Matrix shows 2453 true positives and 11787 true negatives with only 1393 false negatives and 648 false positives:
```
array([[ 2453,  1393],
       [  648, 11787]])
```

other metrics obtained are:

* **Train F1 score** : 0.726
* **Test F1 score** :  0.706

* **Train ROC AUC** : 0.803
* **Test ROC AUC** :  0.793

## Model interpretability

LightGBMs as ensembles of trees are black box algorithms, so in order to know what's happening inside them, we must see which features contribute more information to the decisions of the trees, and that can be seen via the feature importance
![image](https://github.com/nelson-io/citi-documentation-test/raw/main/reports/figures/model_feature_importance.jpg)


We can see that Age, capital gain, years of education, capital loss and hours worked per week are the most important ones which is consistent with business logic. 

#### SHAP

In cases where feature importances are not enough, more complex techniques can be used to bring the information needed from a complex model, for example SHAP values.
the weight each feature contributes to the model output prediction, enlightening data that was previously obscure.


![image](https://github.com/nelson-io/citi-documentation-test/raw/main/reports/figures/model_shap_bs.jpg)

With SHAPS we are able to see the impact of each variable, and furthermore, the role they got in the model.

In the plot above, each dot represents an original observation, so denser areas represent more observations. The dots at the left represent a high negative impact, the ones at the right have a high positive impact and the ones at the middle don't  provide relevant information. At the same time, the color represents high values associated with red and low values with blue, so we can interpret the plot this way:

When marital status is married (high value of the feature) there's a positive impact associated with high income, while low values are associated with low income.

Likewise, we can see that older age is associated with high income while younger age with lower income





