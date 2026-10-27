# Project 2: Housing Price Prediction – Pass 03


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

from sklearn import datasets, metrics
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
target_col = "SalePrice"
feature_types = ["int64"]
```

Create list of target col and feature cols:

```python
# use list comprehension to generate list of target col + selected feature cols
selected_cols = [
    col
    for col in df_preview.columns
    if col == target_col or df_preview[col].dtype in feature_types
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

The ints-feature dataset contains 2637 rows and 27 columns, making for a total of 71199 entries of data. Conveniently, none of these fields is null; as such, no columns or rows need to be removed, and no values need to be imputed.

Details:

  - target column: 'SalePrice'
  - feature columns: ['PID', 'MS SubClass', 'Lot Area', 'Overall Qual', 'Overall Cond',
       'Year Built', 'Year Remod/Add', '1st Flr SF', '2nd Flr SF',
       'Low Qual Fin SF', 'Gr Liv Area', 'Full Bath', 'Half Bath',
       'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces',
       'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch',
       'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold']
  - identifier columns: 'PID'
  - null columns: 0
  - null rows (target): 0
  - null rows (any): 0 (0%)


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
#df again to double-check
df
```

1. Remove identifier columns

```python
# remove identifier columns from DataFrame
df = df.drop( columns = [ 'PID' ] )
```

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

2. Drop null target rows




- No null target rows


3. Drop feature columns with many or all nulls


- No columns with many or all nulls


4. Drop remaining records with null counts (if few)


- No remaining rows with null counts


5.  impute with the median for remaining fields


- No remaining fields are empty, so no imputations necessary

```python
# check again how many of each column is null
df.isnull().sum().sort_values( ascending = False )
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

- For int columns, some features are positively skewed, others negatively skewed, and a couple are evenly distributed
- Lots of positively skewed features (almost look categorical)
- Some features appear categorical (only a few unique values per feature)
- Perhaps some outliers (see positively skewed point above)

```python
# Correlation plot - target and features
plt.figure(figsize=(10,10))
correlation_matrix = df.corr().abs().round(2)
sns.heatmap(data=correlation_matrix, annot=False, vmin=0, vmax=1, square=True) ;


```

```python
# Correlation plot - features only
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

- Not as many non-categorical scatter plots
- some linear-looking trends, including general living area and overall quality (i.e., they highly correlate with the target, and have linear-looking plot clustering)


## Processing



```python

```

## Data Visualization/Communication of Results


```python

```
