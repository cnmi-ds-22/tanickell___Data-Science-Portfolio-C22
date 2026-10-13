# Project 1: Transactions – Pass 08




## Problem Definition




#### 1. State the business problem.

Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.




> We want to predict whether or not a future customer will complete a transaction (e.g., a product purchase, loan or bill payment, or subscription retention) based on prior transaction data. Such predictions can help determine higher-risk customers and present incentives to drive future transaction completions and increase sales.

> This is a supervised, categorical binary classification problem, where the target data is whether a customer transaction was successful (true or false). To achieve this, it uses a Gaussian Naive Bayes model to make predictions about transaction success.





## Data Collection




#### 2. Load Pandas, Numpy, and Matplotlib modules

```python
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
```

#### 3. Load data Train.csv from AWS S3.

```python
# url
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.1-Transactions/Data/Transaction.train.big.csv'
```

- Set number of columns and rows for this pass:

```python
# set cols and rows for pass
num_cols = 50
num_rows = 50_000

num_cols, num_rows
```

- Read in the data from .csv using pandas:

```python
# make dataframe
df = pd.read_csv(
    url,
    usecols=range(num_cols),
    nrows=num_rows,
)
```

- Check the dataframe:

```python
# check dataframe
df
```

## Data Cleaning




#### 4. Examine the data using tools we have used in class. You should be identifying the various columns such as ...

    - target column
    - feature columns
    - identifier columns
    - null columns
    - null rows

    Tools you should be using include, but are not limited to:

    - head
    - tail
    - shape
    - size
    - info
    - describe
    - nunique
    - unique
    - isnull/isna



```python
# define target_col
target_col = 'target'
target_col
```

- head

```python
# head
df.head()
```

- tail

```python
# tail
df.tail()
```

- shape

```python
# shape
df.shape
```

- size

```python
# size
df.size
```

- info

```python
# info
df.info()
```

- describe

```python
# describe
df.describe()
```

- nunique

```python
# nunique
df.nunique()
```

- unique

```python
# unique
df[target_col].unique()
```

- isnull

```python
# isnull
df.isnull().sum().sort_values( ascending = False )
```

```python
# isnull frequencies
(df.isnull().sum() / df.shape[0]).sort_values( ascending = False ) * 100
```

- dtypes

```python
# dtypes
df.dtypes
```

```python
# check to see if anything is not zero -- should return 0 if no nulls
df.isnull().sum().sum()
```

#### 5. NOTE: the ‘target’ column indicates a successful transaction (‘1’) or a no-transaction (‘0’). Verify these are the only values in that column, and that there are no nulls.

```python
# value counts on target
df[target_col].value_counts( dropna = False )
```

```python
# value counts normalized
df[target_col].value_counts( dropna = False, normalize = True ) * 100
```

```python
# target col size/shape
df[[target_col]].shape[0]
```

```python
# sum of nulls in target
df[[target_col]].isnull().sum()
```

```python
# view number of null again
df.isnull().sum()
```

```python
# display columns with all null values
df.isnull().sum()[df.isnull().sum() == 50000]
```

```python
# show columns with non-null values
df.isnull().sum()[df.isnull().sum() != 50000]
```

```python
# show only features
df.isnull().sum()[df.isnull().sum() != 50000][3:]
```

```python
# make list of columns with all nulls
list_nulls = list(df.isnull().sum()[df.isnull().sum() == 50000].keys())
list_nulls
```

```python
# count the number of
len(list_nulls)
```

- Summary:

  - target column: 'target'
  - feature columns: 'var_0' - 'var_46'
  - identifier columns: 'Unnamed', 'ID_code'
  - null columns: 25 (see above)
  - null rows (target): 41494


#### 6. If there are data cleaning issues, develop recommendations for how to deal with them.  Specifically, handle these common issues:

    - identifer columns
    - target column with null counts
    - feature columns with high null counts
    - feature columns with low null counts
    - feature columns with medium null counts


```python
# head again to double-check
df.head()
```

```python
# 1. 'Unnamed' and 'ID_code are very likely identifier columns, and should be dropped; target and var_0 - var_11 will be kept
# 2. 8314 null target columns should be dropped
# 3. 6 null feature columns should be dropped
# 4. After this, drop rows for any column with low null count
# 5. Finally, impute remaining nulls with column medians
```

- backup 1

```python
# make backup
df_bak = df.copy()
df_bak
```

```python
# restore from backup
df = df_bak.copy()
df
```

```python
# keep the target and all features
df = df.drop( columns = ['Unnamed: 0', 'ID_code'] )
```

```python
# check our dataframe (display transformed df)
df
```

- backup 2

```python
# backup 2
df_bak2 = df.copy()
df_bak2
```

```python
# restore from backup 2
df = df_bak2.copy()
df
```

```python
# preview filtering rows for where target is not null
df.loc[df[target_col].isnull() == False]
```

```python
# execute the drop
df = df.loc[df[target_col].isnull() == False].copy()
df
```

```python
# backup 3
df_bak3 = df.copy()
df_bak3
```

```python
# restore from backup
df = df_bak3.copy()
df
```

```python
# review null list
list_nulls
```

```python
# preview drop null columns
df.drop( columns = list_nulls )
```

```python
# drop columns
df = df.drop( columns = list_nulls )
df
```

```python
# make a backup
df_bak4 = df.copy()
df_bak4
```

```python
# restore from backup
df = df_bak4.copy()
df
```

```python
# look for nulls in new cleaned data frame
df.isnull().sum().sort_values( ascending = False )
```

```python
# check total null counts sum
df.isnull().sum().sum()
```

```python
# No more null values, so we're done! ( skip numbers 4 and 5 from above :D )
```

## Exploratory Data Analysis

Produce some visual analysis of the data.  Recall that Gaussian Naive Bayes assumes the features are normally distributed and independent. Note: you might have to do multiple plots in groups.




#### 7. Check the distributions of all variables by plotting.

```python
# Display a histogram
df.hist(figsize=(15, 15))
```

```python
# display a boxplot
df.boxplot(figsize=(15, 15))
```

#### 8. Check the correlation values between all **feature columns** to ensure there are no substantial correlations between features. This is important to support the decision to classify the ‘target’ using Naïve Bayes.



```python
# column correlations
column_correlations = df.corr()
column_correlations
```

```python
# Display correlation matrix
plt.figure(figsize = (25, 25))
sns.heatmap(
    column_correlations,
    annot = True,
    vmin = -1,
    vmax = 1,
    cmap = 'RdYlGn'
)
```

#### 9. Create two data frames: one with all successful transactions, one with all unsuccessful transactions. **Make sure they are copies and not slices**.




```python
# make successful (copy)
df_successful   = df[ df[target_col] == 1 ].copy()
df_successful
```

```python
# make unsuccessful (copy)
df_unsuccessful = df[ df[target_col] == 0 ].copy()
df_unsuccessful
```

## Data Processing




### Single Cross-validation ( CV )




#### 10. Create two data sets: one with all the feature columns (everything except for Unnamed: 0, ID_code and target) and one with just the target. Make sure they are copies and not slices.  And make sure they are the correct dimensions.



```python
# make target and features
target   = df[ target_col ].copy()
features = df.drop( columns = [target_col] ).copy()
```

```python
# check target
target
```

```python
# check target value counts
target.value_counts()
```

```python
# check target dimensions and type
type(target), target.shape, target.ndim
```

```python
# check features
features
```

```python
# check features dimensions and type
type(features), features.shape, features.ndim
```

#### 11. Define an empty Gaussian Naïve Bayes model using Sklearn.



```python
# Create an empty GNB model
modelGNB = GaussianNB()
```

#### 12. Divide the two data sets you created in step #10 into training and testing subsets, i.e. train-test split.



```python
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split( features, target, test_size = 0.25 )
```

#### 13. Train ( fit ) the model using the training subset of the dataset.



```python
# Train ( fit ) the model
modelGNB.fit(X_train, y_train)
```

#### 14. Test the model ( predict ) using the testing subset of the dataset.



```python
# Make predictions using the model
y_pred = modelGNB.predict(X_test)
```

#### 15. Calculate and report the accuracy.



```python
# Calculate the accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy
```

#### 16. Present a confusion matrix and heatmap



```python
# Display confusion matrix
matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(
    matrix.T,
    square = True,
    annot = True,
    fmt = 'd',
    cbar = False
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

#### 17. Present the results of a Classification Report (sklearn.metrics.classification_report).

```python
# 1. Generate the report (as a dictionary, for formatting)
report = metrics.classification_report(y_test, y_pred, output_dict = True, zero_division = 0)

# 2. Convert to DataFrame and transpose
df_report = pd.DataFrame(report).T
df_report
```

```python
# Save variable values
X_train_SCV = X_train
X_test_SCV  = X_test
y_train_SCV = y_train
y_test_SCV  = y_test
y_pred_SCV  = y_pred
```



### Multiple Cross-validation ( CV )



#### 18. Perform a cross-validation loop and record the accuracy of your model during each iteration.



```python
loop_count = 100
```

```python
# Multiple CV loop

# Create array to hold results (accuracies)
n = loop_count
accuracies = np.zeros(n)

# Multiple CV loop
for i in range(n):

  # Train Test Split
  X_train, X_test, y_train, y_test = train_test_split( features, target, test_size = 0.25 )

  # Create an empty GNB model
  modelGNB = GaussianNB()

  # Train ( fit ) the model
  modelGNB.fit(X_train, y_train)

  # Make predictions using the model
  y_pred = modelGNB.predict(X_test)

  # Calculate the accuracy and save to array
  accuracies[i] = metrics.accuracy_score(y_test, y_pred)
```

#### 19. Report the mean accuracy. How does it compare to the accuracy you calculated for the Single CV?



```python
# Calculate mean of results (mean accuracy)
mean_accuracy = accuracies.mean()
mean_accuracy
```

```python
# Save variable values
X_train_MCV = X_train
X_test_MCV  = X_test
y_train_MCV = y_train
y_test_MCV  = y_test
y_pred_MCV  = y_pred
```

Analysis:

- Multiple Cross Validation (Multiple CV) gives a mean accuracy of 91.1% at 100 iterations and 91.1233% at 1000 iterations (so 91.1%).
- This is about the same as the result calculated for the Single CV, which in the previous pass was 91.2%.


#### 20. Plot a histogram of the accuracy scores you generated in your cross-validation loop. What do you notice about the distribution of accuracy scores?



```python
# Display a histogram of the results
pd.Series(accuracies).hist(bins = 30)
```

```python
# --> The distribution of accuracy scores mostly follows a normal distribution.
```

#### 21. Present a confusion matrix and heatmap



```python
# Display a confusion matrix of final result of y_test and y_pred
matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(
    matrix.T,
    square = True,
    annot = True,
    fmt = 'd',
    cbar = False
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

#### 22. Present the results of a Classification Report (sklearn.metrics.classification_report). What do you notice? That is, how do the different metrics compare to each other?



```python
# 1. Generate the report (as a dictionary, for formatting)
report = metrics.classification_report(y_test, y_pred, output_dict = True)

# 2. Convert to DataFrame and transpose
df_report = pd.DataFrame(report).T
df_report
```

Analysis:

- The precision of the model is good for negatives (no transaction) — around 92% — but relatively poor when predicting positives (successful transactions), hovering around 66%. These results worsen when looking at recall (true positives accuracy), which is 98.7% accurate for negatives but only 22.1% accurate for positives. The f1 scores for negatives is similarly high, around 95.2%, and for positives is lower again, at 33.7%.



### Balanced data set

The training data is very skewed towards non-successful transactions (about 90% of the training data has ‘target’==0).




#### 23. Remove enough non-successful transaction rows so that your remaining training data is 50%/50% split between successful and non-successful transactions. Hint: you can use the data frames you created in step #9.



```python
# review value counts
df[target_col].value_counts()
```

```python
# review head of successful
df_successful.head()
```

```python
# review head of unsuccessful
df_unsuccessful.head()
```

- df_unsuccessful is the majority class. Therefore, we can undersample our unsuccessful dataframe as follows:

```python
# preview output of undersampling (downsampling)
df_unsuccessful.sample(df_successful.shape[0]).head()
```

```python
# downsample
df_unsuccessful = df_unsuccessful.sample(df_successful.shape[0])
```

```python
# check it
df_unsuccessful.head()
```

```python
# check sizes to ensure they match
df_successful.shape[0], df_unsuccessful.shape[0]
```

```python
# Combine data frames
df = pd.concat([df_unsuccessful, df_successful], ignore_index = True)
df
```

```python
# check value counts to make sure target successful and unsuccessful counts are equal
df[target_col].value_counts()
```

```python
# make a backup
df_bak3 = df.copy()
df_bak3
```

```python
df = df_bak3.copy()
df
```

```python
# make target and features again
target   = df[ target_col ].copy()
features = df.drop( columns = [target_col] ).copy()
```

```python
target
```

```python
type(target), target.shape, target.ndim
```

```python
features
```

```python
type(features), features.shape, features.ndim
```

#### 24. Repeat the multiple cross-validation process on this data set.



```python
loop_count = 100 # reset the loop count for this loop
```

```python
# Multiple CV loop

# Create array to hold results (accuracies)
n = loop_count
accuracies_balanced = np.zeros(n)

# Multiple CV loop
for i in range(n):

  # Train Test Split
  X_train, X_test, y_train, y_test = train_test_split( features, target, test_size = 0.25 )

  # Create an empty GNB model
  modelGNB = GaussianNB()

  # Train ( fit ) the model
  modelGNB.fit(X_train, y_train)

  # Make predictions using the model
  y_pred = modelGNB.predict(X_test)

  # Calculate the accuracy and save to array
  accuracies_balanced[i] = metrics.accuracy_score(y_test, y_pred)
```

#### 25. Report what your mean accuracy for this balanced data set.



```python
# Calculate mean of results
mean_accuracy_balanced = accuracies_balanced.mean()
mean_accuracy_balanced
```

```python
# Save variable values
X_train_BAL = X_train
X_test_BAL  = X_test
y_train_BAL = y_train
y_test_BAL  = y_test
y_pred_BAL  = y_pred
```

#### 26. Plot a histogram of the accuracy scores you generated in your cross-validation loop. What do you notice about the distribution of accuracy scores?



```python
# Display a histogram of the results
pd.Series(accuracies_balanced).hist(bins = 30)
```

#### 27. Present a confusion matrix and heatmap



```python
# Display a confusion matrix of final result of y_test and y_pred
matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(
    matrix.T,
    square = True,
    annot = True,
    fmt = 'd',
    cbar = False
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

#### 28. Present the results of a Classification Report (sklearn.metrics.classification_report). What do you notice? That is, how do the different metrics compare to each other?



```python
# 1. Generate the report (as a dictionary, for formatting)
report = metrics.classification_report(y_test, y_pred, output_dict = True)

# 2. Convert to DataFrame and transpose
df_report = pd.DataFrame(report).T
df_report
```

## Data Visualization




#### 29. Compare the accuracy results of your multiple cross-validation with the whole training data and the balanced training data



```python
accuracy
```

```python
float(mean_accuracy)
```

```python
float(mean_accuracy_balanced)
```

#### 30. Present the confusion matrix and heatmap



```python
# Display a confusion matrix of final result of y_test and y_pred
matrix = metrics.confusion_matrix(y_test_SCV, y_pred_SCV)
sns.heatmap(
    matrix.T,
    square = True,
    annot = True,
    fmt = 'd',
    cbar = False
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

```python
# Display a confusion matrix of final result of y_test and y_pred
matrix = metrics.confusion_matrix(y_test_MCV, y_pred_MCV)
sns.heatmap(
    matrix.T,
    square = True,
    annot = True,
    fmt = 'd',
    cbar = False
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

```python
# Display a confusion matrix of final result of y_test and y_pred
matrix = metrics.confusion_matrix(y_test_BAL, y_pred_BAL)
sns.heatmap(
    matrix.T,
    square = True,
    annot = True,
    fmt = 'd',
    cbar = False
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

#### 31. Present the results of your Classification Report (sklearn.metrics.classification_report)




```python
# 1. Generate the report (as a dictionary, for formatting)
report = metrics.classification_report(y_test_SCV, y_pred_SCV, output_dict = True)

# 2. Convert to DataFrame and transpose
df_report = pd.DataFrame(report).T
df_report
```

```python
# 1. Generate the report (as a dictionary, for formatting)
report = metrics.classification_report(y_test_MCV, y_pred_MCV, output_dict = True)

# 2. Convert to DataFrame and transpose
df_report = pd.DataFrame(report).T
df_report
```

```python
# 1. Generate the report (as a dictionary, for formatting)
report = metrics.classification_report(y_test_BAL, y_pred_BAL, output_dict = True)

# 2. Convert to DataFrame and transpose
df_report = pd.DataFrame(report).T
df_report
```

## Communicate the Results




#### 32. Communicate the results of your analysis.



Conclusions (Brief):

- The mean accuracy for the balanced set was considerably lower than the accuracy and mean accuracy for the SingleCV and MultipleCV sets, at around 75%.
- The precision of true negatives likewise decreased from around 91% to 75%.
- However, the accuracy of prediction for true negatives has gone up, from around 66% to 75%.
- Even more significant, the recall of positives (ratio of true positives to total positives) has gone way up, from 22% to 75%.
- Percentage accuracy, recall, and f1-score of false positives, false negatives, true positives, and true negatives all about the same as each other.
- Much more balanced (and informative) model.



Future Work:
- bigger data set
- other types of validation?
- other types of data set balancing (upsampling?)
- different model


## Submit Final Project

#### 33. Save ( commit ) your finished Jupyter notebook to your Project 1 GitHub repo.

