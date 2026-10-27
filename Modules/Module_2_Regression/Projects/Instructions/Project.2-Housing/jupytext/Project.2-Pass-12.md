# Project 2: Housing Price Prediction – Pass 12


## Problem Definition



State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.

Statement:

We want to predict housing sale prices based on various correlated feature data. This is a *supervised regression* problem, where the target of the prediction is sales prices. To accomplish this, we use a regression model alongside Regression prediction techniques, as well as feature scaling and regularization to minimize the RMS percentage error.


## Data Collection/Sources



Data Collection describes the process by which we obtain the data for use in the project. In this case, our data comes from S3 storage buckets on AWS, which we load into memory (variables) via URLs by passing them to the read_csv function of the pandas python library (more on that soon).

The data comes in one large set containing the full data available for use in the project. It is supplemented by a data dictionary that describes the set, the data types of the included data, and other information relevant to its use.

Finally, to store the data in a form we can work with in memory, we use a storage structure called a DataFrame object from the pandas python library. This object allows us to quickly view the data and preview operations on the data without changing it, as well as make updates to the data in a highly intuitive way.


First, load Pandas, Numpy, and Matplotlib modules:

```python
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from sklearn import datasets, metrics, model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
```

Load data Train.csv from AWS S3:

```python
# url
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Data/Housing.Data.csv'
```

Next, make a preview DataFrame to look at columns:

```python
# make preview
df_preview = pd.read_csv(url, nrows=1)
df_preview
```

```python
# preview all column names
df_preview.columns
```

Define the target column (and features types, if required):

```python
# define target column (by name) and target features type
target_col = 'SalePrice'
feature_types = ['float64', 'int64']
```

Create list of target col and feature cols:

```python
# use list comprehension to generate list of target col + selected feature cols
selected_cols = [
    col
    for col in df_preview.columns
    if (col == target_col) or (df_preview[col].dtype in feature_types)
]
selected_cols
```

Read in the data from .csv using target_col and feature_types:

```python
# make dataframe
df = pd.read_csv( url, usecols=selected_cols )
```

Finally, check the dataframe:

```python
# check dataframe
df
```

```python
# use list comprehension to generate list of target col + selected feature cols
selected_cols = [
    col
    for col in df.columns
    if (col == target_col) or (df[col].dtype in feature_types)
]
selected_cols
```

```python
# make dataframe
df = pd.read_csv( url, usecols=selected_cols )
```

```python
# check dataframe
df
```

## Data Cleaning



Data Cleaning is the process of preparing the data set for use in analysis. Many data sets contain bad or null (empty) values, which can cause errors when trying to work with the data and use it in code. Cleaning the data involves removing bad (null-containing) columns and/or rows from the set, as well as imputing (replacing via insertion) similar data into fields with missing or unusable values.

Here, to clean the data, we mostly use the pandas drop function included with pandas DataFrame objects, which allow us to drop columns and rows according to conditions we specify, much like SQL WHERE clauses.


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
df.nunique().sort_values( ascending = False )
```

```python
# normalize to look for ID column(s)
df.nunique().sort_values( ascending = False ) / df.shape[0] * 100
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
# percent null by column
(df.isnull().sum() / df.shape[0]).sort_values( ascending = False ) * 100
```

```python
# look at unique values in the col-by-col null counts to classify cols
((df.isnull().sum() / df.shape[0]).sort_values( ascending = False ) * 100).unique()
```

```python
# check to see if anything is not zero -- should return 0 if no nulls
df.isnull().sum().sum()
```

```python
# how many rows with at least one null
num_rows_with_nulls = df.isna().any(axis=1).sum()
int(num_rows_with_nulls)
```

Many different null counts exist throughout the data set, indicating that this will not be a straightforward nulls cleanup.


- dtypes

```python
# dtypes
df.dtypes
```

Examine null columns, columns with some nulls, and columns with no nulls:

```python
# display columns with all null values
df.isnull().sum()[df.isnull().sum() == df.shape[0]]
```

```python
# count these
int(df.isnull().sum()[df.isnull().sum() == df.shape[0]].count())
```

```python
# show columns with some null values
df.isnull().sum()[(df.isnull().sum() != df.shape[0]) & (df.isnull().sum() != 0)].sort_values( ascending = False )
```

```python
# count these
int(df.isnull().sum()[(df.isnull().sum() != df.shape[0]) & (df.isnull().sum() != 0)].count())
```

```python
# columns with no nulls
df.isnull().sum()[df.isnull().sum() == 0]
```

```python
# count these
int(df.isnull().sum()[df.isnull().sum() == 0].count())
```

#### Target: Value Counts

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
int(df[[target_col]].isnull().sum().iloc[0])
```

```python
# sum of non-nulls in target
df[[target_col]].shape[0] - int(df[[target_col]].isnull().sum().iloc[0])
```

There are no nulls in the target column, which means no rows will need to be dropped on its account.


#### Summary:

The floats+ints-feature dataset contains 2637 rows and 38 columns, making for a total of 100206 entries of data. Many of these fields are null.

Details:

  - target column: 'SalePrice'
  - feature columns: [
       'Lot Frontage', 'Garage Yr Blt', 'Mas Vnr Area', 'Bsmt Full Bath',
       'Bsmt Half Bath', 'BsmtFin SF 2', 'BsmtFin SF 1', 'Total Bsmt SF',
       'Bsmt Unf SF', 'Garage Area', 'Garage Cars',
       'MS SubClass', 'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold'
       ]
  - identifier columns: 'PID'
  - null columns: 0
  - null rows (target): 0
  - null rows (any): 2637 (100%)


#### If there are data cleaning issues, develop recommendations for how to deal with them.  Specifically, handle these common issues:

    - identifer columns
    - target column with null counts
    - feature columns with high null counts
    - feature columns with low null counts
    - feature columns with medium null counts



First, make a backup:

```python
# make backup
df_bak = df.copy()
```

```python
# restore from backup
df = df_bak.copy()
```

```python
# df again to double-check
df
```

1. Remove identifier columns

```python
# remove identifier columns from DataFrame
df = df.drop( columns = [ 'PID' ] )
```

```python
# backup 1
df_bak1 = df.copy()
```

```python
# restore from backup 1
df = df_bak1.copy()
```

```python
df
```

2. Drop null target rows




- No null target rows


3. Drop feature columns with many or all nulls

```python
df.isnull().sum().sort_values( ascending = False )
```

```python
# df = df.loc[:, df.isnull().mean() <= 0.5]
df = df.drop( columns = [index for index in list(df.isnull().sum().index) if (df.isnull().sum().loc[index] / df.shape[0] > 0.4)])
```

```python
df
```

```python
list(df.isnull().sum().index)
```

```python
df.shape
```

4. Drop remaining records with null counts (if few)

```python
# backup 2
df_bak2 = df.copy()
```

```python
# restore from backup 2
df = df_bak2.copy()
```

```python
df
```

```python
df.isnull().sum().sort_values( ascending = False )
```

```python
df = df.dropna(subset=[index for index in list(df.isnull().sum().index) if (df.isnull().sum().loc[index] / df.shape[0] < 0.05)])
```

```python
df
```

```python
list(df.isnull().sum().index)
```

```python
df.shape
```

5.  impute with the median for remaining fields

```python
# backup 3
df_bak3 = df.copy()
```

```python
# restore from backup 3
df = df_bak3.copy()
```

```python
df
```

```python
# check again how many of each column is null
df.isnull().sum().sort_values( ascending = False )
```

```python
# normalize null counts (per column)
df.isnull().sum().sort_values( ascending = False ) / df.shape[0] * 100
```

Impute:

```python
feature_cols = list(df.drop(columns = target_col).columns)
feature_cols
```

```python
# loop through features to bulk impute with respective column medians
for feature_col in feature_cols:

  # Calculate the median
  median = df[feature_col].median()
  print(median)

  # Assign median to nulls
  df.fillna( {feature_col: median}, inplace = True )

  # Verify median is about the same as before
  print(df[feature_col].median())
```

```python
df
```

```python
# check again how many of each column is null
df.isnull().sum().sort_values( ascending = False )
```

```python
# backup 4
df_bak4 = df.copy()
```

```python
# restore from backup 4
df = df_bak4.copy()
```

```python
df
```

```python
# check total null counts sum
f"Number of remaining nulls in df: {int(df.isnull().sum().sum())}"
```

## Exploratory Data Analysis


```python
len(df.columns)
```

```python
# Let's plot mpg and all of our predictors
subplots_per_row = 4
total_subplots = len(df.columns)
row_count = total_subplots // subplots_per_row
fig, subplots = plt.subplots(row_count,subplots_per_row, figsize = (20,4*row_count)) # Creating 2x4 subplots because we have 8 columns in our data
column_names = df.columns

subplots = subplots.flatten()[:len(column_names)]  # flatten and slice axs
for n, subplot in enumerate(subplots):             # Loop through axs
  subplot.hist(df[column_names[n]])
  subplot.set_xlabel(column_names[n])
  subplot.set_ylabel('Count')
```

### What do we notice?

- SalePrice and Lot Frontage are positively skewed
- All float features look positively skewed
- Some features appear categorical (only a few unique values per feature)
- Perhaps some outliers
- For int columns, some features are positively skewed, others negatively skewed, and a couple are evenly distributed
- Lots of positively skewed features (almost look categorical)
- Some features appear categorical (only a few unique values per feature)
- Perhaps some outliers (see positively skewed point above)

```python
# Correlation plot — target and features
plt.figure(figsize=(10,10))
correlation_matrix = df.corr().abs().round(2)
sns.heatmap(data=correlation_matrix, annot=False, vmin=0, vmax=1, square=True) ;


```

```python
# Correlation plot — features only
plt.figure(figsize=(10,10))
correlation_matrix = df.drop( columns = [target_col] ).corr().abs().round(2)
sns.heatmap(data=correlation_matrix, annot=False, vmin=0, vmax=1, square=True) ;


```

```python
correlation_matrix = df.corr().round(2)
correlation_matrix[target_col].abs().sort_values(ascending=False)
```

```python
correlation_matrix = df.corr().round(2)
correlation_matrix[target_col].abs().sort_values(ascending=False).plot( kind="bar", ylim = [0,1])
```

```python
corrs = df.corr().round(2).unstack().abs()
corrs.reset_index()
```

```python
corrs = corrs[corrs < 1]
corrs.reset_index()
```

```python
corrs.sort_values(ascending = False).reset_index()
```

```python
corrs.sort_values(ascending = False)[::2].reset_index()
```

### What do we notice?

- SalePrice vs. Lot Frontage do not appear to be highly correlated to each other
- BsmtFin SF 2 and Bsmt Half Bath are very independent from most other features, but also poorly coorelate with SalePrice
- Garage Area and Garage Cars coorelate with target well and each other fairly well, but don't correlate with other features nearly as much, meaning one of these could be a good candidate

- SalePrice correlated with General living area, first floor sf, year built, and overall quality
- lattice pattern: typically, if a feature doesn't correlate with other features, it also doesn't correlate with the target

```python
df.size
```

```python
# Let's plot a pairplot
sns.pairplot(df) ;


```

### What do we notice?

- SalePrice vs. Lot Frontage is rather scattered, and does not appear to be linear.
- Many scattered relationships, some with some degree of potential linearity
- Not as many non-categorical scatter plots
- some linear-looking trends, including general living area and overall quality (i.e., they highly correlate with the target, and have linear-looking plot clustering)


## Processing




### Prepare data frame

```python
# Correlation plot — target and features (again)
plt.figure(figsize=(10,10))
correlation_matrix = df.corr().abs().round(2)
sns.heatmap(data=correlation_matrix, annot=False, vmin=0, vmax=1, square=True) ;


```

Here, we can make a few notes about the best features:
- BsmtFin SF 2 and Bsmt Half Bath are quite non-correlated with SalePrice (0.01 and 0.04, respectively), so we can safely exclude them
- Bsmt Unf SF, Bsmt Full Bath, and Lot Frontage are all weakly correlated (0.18, 0.28, and 0.34), but should be kept for now
- Garage Cars and Garage Area are strongly correlated with each other, so only one of these should be kept
- Everything else looks sufficiently independent that they may be kept

```python
# use threshold value to drop features that are statistical noise
threshold_val = 0.05
drop_cols_threshold = [col for col in df.columns if correlation_matrix[target_col].abs().loc[col] < threshold_val]
```

```python
# list of redundants
redundants = ['Garage Area', 'TotRms AbvGrd', 'Garage Yr Blt', 'Total Bsmt SF']
```

```python
target_col = 'SalePrice'
drop_cols = drop_cols_threshold + redundants + [target_col]
```

```python
target_col = 'SalePrice'
feature_cols = list( df.drop( columns = drop_cols ) )
target_col, feature_cols, drop_cols
```

```python
# Split data frame into target and features
target   = df[target_col].copy()
features = df[feature_cols].copy()
```

```python
# Verify features
features
```

```python
# Verify target
target
```

```python
type(features), features.shape, features.ndim
```

```python
type(target), target.shape, target.ndim
```

### Fit & Assess Our Model


#### Ridge Regression — Sklearn


Now, we perform a ridge regression using sklearn:

```python
# First, we scale the values with a scaler:
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
```

```python
# Run a Single Split to test the model:
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size = 0.2, random_state=42)
```

```python
# Fit a ridge model using Sklearn
model = Ridge(alpha = 1)
model.fit(X_train, y_train)

( model.intercept_, model.coef_ )
```

```python
# See how well our model performs on our test data
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.plot([min(y_pred), max(y_pred)],[min(y_pred), max(y_pred)], c='red')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price') ;
```

```python
list(zip(y_test,y_pred))[:10]
```

We'll use RMSE as our performance metric.

```python
# Calculate RMSE
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean(np.square(y_test - y_pred)))
print(f"RMSE: {rmse}")
```

```python
# Check root mean squared error on test data
rmse = metrics.mean_squared_error(y_test, y_pred)**0.5
print(f"RMSE: {rmse}")
```

Perform the Regression:

```python
# Specify values of alpha we want to try
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 10, 20, 30, 50]

# Create empty list to hold results
results_alpha = []

for a in alpha:
  # Train the model
  my_fit_rr = Ridge(alpha = a) # Looping through alpha
  my_fit_rr.fit(X_train, y_train)

  # Calculate CV RMSE
  results = cross_val_score(
      my_fit_rr,
      features_scaled,
      target,
      scoring='neg_root_mean_squared_error',
      cv = 10
      )
  rmse = abs(results.mean())
  results_alpha.append(rmse)

results_df = pd.DataFrame({'Alpha': alpha, 'RMSE': results_alpha})
print(results_df.sort_values(by=['RMSE']))
```

```python
results_df.sort_values( by = ["Alpha"])
```

```python
results_df.sort_values( by = ["Alpha"])["RMSE"].plot()
```

```python
# Retrain the model w/ alpha = 20
my_fit_rr = Ridge(alpha = 20)
my_fit_rr.fit(X_train, y_train)

# Calculate CV RMSE
results = cross_val_score(my_fit_rr, features_scaled, target, scoring='neg_root_mean_squared_error', cv = 10)
rmse = abs(results.mean())
print(f"CV RMSE: {rmse}")
```

## Data Visualization/Communication of Results



### Residual Analysis

```python
# QQ Plot - Residuals are normally distributed
res = y_test - y_pred
sm.qqplot(res, fit=True, line="45") ;
```

```python
# Residuals vs Fitted (Predicted) Values - Constant variance
plt.figure(figsize = (10,6))
plt.scatter(y_pred, res)
plt.xlabel("Fitted")
plt.ylabel("Residuals")
plt.hlines(0, min(y_pred), max(y_pred), colors = 'red', linestyles = 'dashed') ;
```

```python
# Residuals vs Time - indepedence
plt.figure(figsize = (10,6))
plt.scatter(range(len(res)), res)
plt.plot(range(len(res)), res, 'b')
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.hlines(0, min(range(len(res))), max(range(len(res))), colors = 'red', linestyles = 'dashed') ;
```

This result tells us that altogether, using multiple float features produces a better model for house sale price, better than using Garage Area on its own. However, the residual analysis still shows a non-normal distribution in the residuals, showing that something is awry in the fit between the data and the model, implying linear regression may not be properly modeling the relationship between the features and the target.


## Data Visualization/Communication of Results


```python

```
