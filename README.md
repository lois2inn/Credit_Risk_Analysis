# Credit_Risk_Analysis

Apply Machine Learning models to predict credit risk.

## Overview

Fast Lending, a peer to peer lending services company wants to use Machine Learning to predict credit risk. The management believes that this will provide a quicker and more reliable loan experience leading to a more accurate identification of good candidates for loans which will result in lower default rates. The project aims to build and evaluate several machine learning models to predict credit risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Techniques like training, resampling and boosting to make the most of the models and credit card data are employed all along. 
- Oversample the data using the RandomOverSampler and SMOTE algorithms.
- Undersample the data using the ClusterCentroids algorithm.
- Use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm.
- Compare BalancedRandomForestClassifier and EasyEnsembleClassifier models to reduce bias and predict credit risk.
- Evaluate the performance of each of the models and make a recommendation on whether they should be used to predict credit risk.

## Resources

Anaconda 2022.10
ipykernel 6.15.2
Jupyter Notebook
Python 3.7.13
Pandas 1.3.5
Scikit-learn 1.0.2
imbalanced-learn 0.7.0
Dataset from LendingClub 

## Results
Across all the models compared, 
the training and testing sets are split in 75% and 25% of total data respectively
random_state=1

Describe the balanced accuracy score and the precision and recall scores of all six machine learning models.

- evaluate three machine learning models by using resampling to determine which is better at predicting credit risk.
  - Random Oversampler
  - SMOTE
  - Cluster centroids
  
- use SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the above resampling algorithms 

- Compare classifiers using BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model.

## Summary

- summary of the results
- a recommendation on which model to use, or there is no recommendation with a justification

