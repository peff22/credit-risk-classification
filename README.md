## Credit Risk Analysis Report

### Overview of the Analysis

The purpose of this analysis is to create a data model that accurately predicts the credity worthiness of potential borrowers from peer-to-peer lending services.

The dataset used in this analysis is found in "lending_data.csv". The dataset offers historical data pertaining to lending activity from a peer to peer lending services company. 
I used this data to build a model that predicts the creditworthiness of borrowers based on the following factors:
loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt, and loan status. 

To estimate creditworthiness, I stored the labels set from the loan_status column in the y variable. Then, I stored the features DataFrame (all the columns except loan_status)
in the X variable. I checked the balance of the labels with value_counts. The results showed that, in this dataset, 75036 loans were healthy and 2500 were high-risk.

I used the train_test_split module from sklearn to split the data into training and testing variables, these are: 
X_train, X_test, y_train, and y_test. And I assigned a random_state of 1 to the function to ensure that the train/test split is consistent, 
the same data points are assigned to the training and testing sets across multiple runs of code.

I created a Logistic Regression Model with the original data. I used LogisticRegression(), from sklearn, 
with a random_state of 1. I fit the model with the training data, X_train and y_train, 
and predicted on testing data labels with predict() using the testing feature data, X_test, and the fitted model, lr_model.

I calculated the accuracy score of the model with balanced_accuracy_score() from sklearn, I used y_test a d testing_prediction to obtain the accuracy.

I generated a confusion matrix for the model with confusion_matrix() from sklearn, based on y_test and testing_prediction.

I obtained a classification report for the model with classification_report() from sklearn, and I used y_test and testing_prediction.

I used RandomOverSampler() from imbalanced-learn to resample the data. I fit the model with the training data, X_train and y_train. 
I generated resampled data, X_resampled and y_resampled, and used unique() to obtain the count of distinct values in the resampled labels data.

Then, I created a Logistic Regression Model with the resampled data, fit the data, and made predictions. 
Lastly, I obtained the accuracy score, confusion matrix, and classification report of the resampled model.


## Results

* Machine Learning Model 1:

  *Model 1 Accuracy: 0.952.
  *Model 1 Precision: for healthy loans the precision is 1.00, for high-risk loans the precision is 0.85.
  *Model 1 Recall: for healthy loans the recall score is 0.99, for high-risk loans the recall score is 0.91.



* Machine Learning Model 2:

  *Model 2 Accuracy: 0.995.
  *Model 2 Precision: for healthy loans the precision is 0.99, for high-risk loans the precision is 0.99.
  *Model 2 Recall: for healthy loans the recall score is 0.99, for high-risk loans the recall score is 0.99.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Each model peforms differently, so the context is really important in trying to determine which model performs "best".
* The first model seems to be better suited for predicting healthy loans, so if that is your goal then I would recommend that model.
* The second model performs with much higher accuracy than the first model when it comes to predicting high risk loans, so I would recommend the second model if that is your goal.
* I would imagine this company is more concerned with high risk loans, so I would recommend the second model for their use.

