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
- The target of the data set, "loan status", is used to determine whether the credit application is considered "low" or "high" risk. Applications that have "current" as the "loan status" are classified as "low risk" and the remaining as "high risk". With this consideration, the dataset has a total of 68,817 applications with 99% classified as "low risk".
<img src="ml_challenge/images/target_cnt.png" width="300"/>

- The training and testing sets are split in 75% and 25% of total data respectively. This shows that 51,352 "low risk" and 240 "high risk" applications are categorized into the training set.
<img src="ml_challenge/images/train_test.png" width="550"/>

- The data is oversampled using the Naive Random Oversampling algorithm and the SMOTE algorithm.

#### Naive Random Oversampling 
- This model randomly selects samples (with replacement) from the minority class and adds to the training set until both minority and majority classes are balanced. As a result, both high-risk and low-risk categories count at 51,352.
<img src="ml_challenge/images/ros_set.png" width="400"/>

- The resampled data is then used to train a logistic regression model. The metrics generated are shown below:
<img src="ml_challenge/images/model_1.png" width="600"/>

- The balanced accuracy score reads 65%. 
- There are 55 instances of "High Risk" class that are predicted correctly while 32 of "Low Risk" class are predicted falsely as "High Risk". Also, 5570 "Low Risk" applications are falsely predicted as "High Risk". The model shows a slightly better performance in identifying “Low Risk” class than a “High Risk” class. 
- The "High Risk" precision rate is 1% which means that only 1% of predicted "High Risk" applications are truly "High Risk".
- The "High Risk" recall registers 63%. That means among all predicted 5620 "High Risk" applications, only 63% are correctly predicted giving this model an F1 score of 2%.
- The "Low Risk" has a precision rate of 100% and recall at 67%.

#### SMOTE (Synthetic Minority Oversampling Technique)
- Like Random Oversampling technique, SMOTE also increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection. The resampled data in minority and majority classes count at 51,352.
- Upon training a logistic regression model using the resampled data, the following metrics are generated:
<img src="ml_challenge/images/model_2.png" width="600"/>

- The balanced accuracy score has recorded 65% which is the same as Naive Random Oversampler.
- Like RandomOverSampler, the "High Risk" precision rate again is only 1% with the recall improve to 64% giving this model an F1 score of 2%.
- "Low Risk" class has a precision rate of 100% and decrease in recall at 66%.


#### Undersampling with Cluster Centroids algorithm
-
<img src="ml_challenge/images/model_3.png" width="600"/>

#### Combination Sampling using SMOTEENN algorithm
-
<img src="ml_challenge/images/model_4.png" width="600"/>

#### Balanced Random Forest Classifier
- 
<img src="ml_challenge/images/model_5.png" width="600"/>

#### Easy Ensemble AdaBoost Classifier
-
<img src="ml_challenge/images/model_6.png" width="600"/>

- Precision is more than recall, this should be the case for loan classifiers in banking. Higher precision classifiers won’t accidentally reject a potential customer and lose the business.

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

