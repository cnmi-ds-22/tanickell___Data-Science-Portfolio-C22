# Project 1: Transactions – Pass 01




## Problem Definition




#### 1. State the business problem.

Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.




> We want to predict whether or not a future customer will complete a transaction (e.g., a product purchase, loan or bill payment, or subscription retention) based on prior transaction data. Such predictions can help determine higher-risk customers and present incentives to drive future transaction completions and increase sales.

> This is a supervised, categorical binary classification problem, where the target data is whether a customer transaction was successful (true or false). To achieve this, it uses a Gaussian Naive Bayes model to make predictions about transaction success.




## Data Collection




#### 2. Load Pandas, Numpy, and Matplotlib modules

```python
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
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.1-Transactions/Data/Transaction.train.csv'
```

- Set number of columns and rows for this pass:

```python
num_cols = 5
num_rows = 1_000

num_cols, num_rows
```

- Read in the data from .csv using pandas:

```python
df = pd.read_csv(
    url,
    usecols=range(num_cols),
    nrows=num_rows,
)
```

- Check the dataframe:

```python
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
df.head()
```

- tail

```python
df.tail()
```

- shape

```python
df.shape
```

- size

```python
df.size
```

- info

```python
df.info()
```

- describe

```python
df.describe()
```

- nunique

```python
df.nunique()
```

- unique

```python
df[target_col].unique()
```

- isnull

```python
df.isnull().sum().sort_values( ascending = False )
```

```python
(df.isnull().sum() / df.shape[0]).sort_values( ascending = False ) * 100
```

- dtypes

```python
df.dtypes
```

#### 5. NOTE: the ‘target’ column indicates a successful transaction (‘1’) or a no-transaction (‘0’). Verify these are the only values in that column, and that there are no nulls.

```python
df[target_col].value_counts( dropna = False )
```

```python
df[[target_col]].shape[0]
```

```python
df[[target_col]].isnull().sum()
```

- Summary:

  - target column: 'target'
  - feature columns: 'var_0', 'var_1'
  - identifier columns (likely): 'Unnamed', 'ID_code'
  - null columns: n/a
  - null rows: n/a

  No null values at all this pass.


#### 6. If there are data cleaning issues, develop recommendations for how to deal with them.  Specifically, handle these common issues:

    - identifer columns
    - target column with null counts
    - feature columns with high null counts
    - feature columns with low null counts
    - feature columns with medium null counts


```python
df.head()
```

```python
# 'Unnamed' and 'ID_code are very likely identifier columns, and should be dropped

# target, var_0, and var_1 are meaningful data columns and will be kept
```

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

- This pass, keep only the target and two features:

```python
target_cols  = [target_col]         # defined as 'target' above
feature_cols = ['var_0', 'var_1']
df = df[target_cols + feature_cols]
```

```python
# check our dataframe (display transformed df)
df
```

- backup 2

```python
df_bak2 = df.copy()
df_bak2
```

```python
df = df_bak2.copy()
df
```

```python
# No rows have nulls, so we can skip imputing with fillna() and/or dropping rows
```

## Exploratory Data Analysis

Produce some visual analysis of the data.  Recall that Gaussian Naive Bayes assumes the features are normally distributed and independent. Note: you might have to do multiple plots in groups.




#### 7. Check the distributions of all variables by plotting.

```python
# Display a histogram
df.hist()
```

```python
df.boxplot()
```

#### 8. Check the correlation values between all **feature columns** to ensure there are no substantial correlations between features. This is important to support the decision to classify the ‘target’ using Naïve Bayes.



```python
column_correlations = df.corr()
column_correlations
```

```python
# Display correlation matrix
plt.figure(figsize = (10, 6))
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
df_successful   = df[ df[target_col] == 1 ].copy()
df_successful
```

```python
df_unsuccessful = df[ df[target_col] == 0 ].copy()
df_unsuccessful
```

## Data Processing




### Single Cross-validation ( CV )




#### 10. Create two data sets: one with all the feature columns (everything except for Unnamed: 0, ID_code and target) and one with just the target. Make sure they are copies and not slices.  And make sure they are the correct dimensions.



```python
target   = df[ target_col ].copy()
features = df.drop( columns = [target_col] ).copy()
```

```python
target
```

```python
target.value_counts()
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
# Calculate the accuracy and save to array
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
