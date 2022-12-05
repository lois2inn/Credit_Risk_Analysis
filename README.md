# Credit_Risk_Analysis

Apply Machine Learning models to predict credit risk.

## Overview

Fast Lending, a peer to peer lending services company wants to use Machine Learning to predict credit risk. The management believes that this will provide a quicker and more reliable loan experience leading to a more accurate identification of good candidates for loans which will result in lower default rates. The project aims to build and evaluate several machine learning models to predict credit risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Techniques like training, resampling and boosting are employed to make the most of the models and credit card data. 
- Oversample the data using the RandomOverSampler and SMOTE algorithms.
- Undersample the data using the Cluster Centroids algorithm.
- Use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm.
- Compare Balanced Random Forest Classifier and Easy Ensemble Classifier models to reduce bias and predict credit risk.
- Evaluate the performance of each of the models and make a recommendation on whether they should be used to predict credit risk.

## Resources

- Anaconda 2022.10
- ipykernel 6.15.2
- Jupyter Notebook
- Python 3.7.13
- Pandas 1.3.5
- Scikit-learn 1.0.2
- imbalanced-learn 0.7.0
- Dataset from LendingClub 

## Results

- Scikit-learn and imbalanced-learn Python libraries are used resample the dataset and evaluate results.
- To ensure consistency between tests, a random_state of 1 is used for each sampling algorithm.
- The target of the data set, "loan status", is used to determine whether the credit application was considered "low" or "high" risk. Applications that have "current" as the "loan status" are classified as "low risk" and the remaining as "high risk". With this consideration, the dataset has a total of 68,817 applications with 99% classified as "low risk".
<img src="images/target_cnt.png" width="400"/>

- The training and testing sets are split in 75% and 25% of total data respectively.
<img src="images/train_test.png" width="400"/>

<table>
 <tr>
  <td>
    <img src="images/model_1.png"/>
   </td>
 </tr>
</table>

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

