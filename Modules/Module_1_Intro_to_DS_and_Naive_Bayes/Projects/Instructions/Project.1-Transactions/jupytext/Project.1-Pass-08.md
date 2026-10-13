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

Conclusions (Brief):

- The accuracy for this larger subset of the large data set was about the same but a little lower, at about exactly 90%. Again, this is about as good as random, in that it matches the 90-10 unsuccessful-to-successful split.
- Like with the smaller data set, the number of true positives rose a bit with the additional data, to 10 / 2127 (where before it was 0).
- The number of false positives also rose to 7.
- The number of false negatives proportionally remained about the same, moving just a bit from 46 / 422, just over 10%, to 205 / 2127, just under 10%, which is a marginal improvement.



Future Work:
- bigger data subset (or set?)
- other types of validation?
- other types of data set balancing (upsampling?)
- different model
