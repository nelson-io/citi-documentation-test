# Model Optimization Report

The optimization of this model was made with [Optuna](https://optuna.org/), An open source hyperparameter optimization framework to automate hyperparameter search.

The model used in this project was made with [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/index.html), a gradient boosting framework that uses tree based learning algorithms.

As with other complex frameworks, when training a model with LightGBM, there are a set of parameters that are learnt by the model during the training stage while there are other parameters (Hyperparameters) that must be set before initiating the algorithm.

**LightGBM Hyperparameters**: 
```
                boosting_type
                lambda_l1
                lambda_l2
                num_leaves
                feature_fraction
                bagging_fraction
                bagging_freq
                min_child_samples
```

The strategy opted for the optimization of this model, was to conduct a study with 300 iterations of the model. 

In each iteration, the training data was alocated into 5 folds and there were used 4 folds were used to train an algorithm while obtaining the **Accuracy** on the remaining fold. This process was repeated for every combination of folds, afterwards,  averaging the accuracy results, weighting them to grade how the model did. This process is called K-fold cross-validation.

![image](https://www.philschmid.de/static/blog/k-fold-as-cross-validation-with-a-bert-text-classification-example/k-fold.svg)

The first 100 iterations had values selected at random (Random Search) while the remaining 200 were determined with bayesian optimization harnessing the information revealed by previous iterations in order to recommend better hyperparameter combinations.

As a result, the following visualization shows the history of how the accuracy improved as the study progressed with each iteration:
![image](https://github.com/nelson-io/citi-documentation-test/raw/main/reports/figures/optuna_history.png)

It is stated unequivocally the random behaviour of the first phase and the improvement evidenced with each run until achieving the highest  performance in iteration 156 with an average accuracy of 0.8781.

As the study became to a conclusion, the best hyperparameters were extracted to re-train the model with all the training set.

The best Hyperparameters obtained were:
```
{'lambda_l1': 2.3859884931142832e-05,
 'lambda_l2': 0.07560475953921798,
 'num_leaves': 27,
 'feature_fraction': 0.41216366629200235,
 'bagging_fraction': 0.7610827990126722,
 'bagging_freq': 6,
 'min_child_samples': 13}
```

Nevertheless, not every Hyperparameter had the same impact in the final model. The number of leaves was the most significant hyperparameter to tune followed by the feature_fraction, the bagging fractions and the minimum child samples.

![image](https://github.com/nelson-io/citi-documentation-test/raw/main/reports/figures/optuna_importance.png)

Other insight that arises from this analysis, is that as the results converge so do the hyperparameter combinations, showing that the best ones tend to have a big bagging fraction, bagging frequency and lambda L2 while low values for feature fraction, minimum child samples and number of leaves.

![image](https://github.com/nelson-io/citi-documentation-test/raw/main/reports/figures/optuna_parallelcoordinate.png)