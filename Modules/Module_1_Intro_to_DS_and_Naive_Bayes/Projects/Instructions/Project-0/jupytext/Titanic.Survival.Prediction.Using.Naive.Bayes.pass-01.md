# Naive Bayes Classification to Predict Titanic Survival


**Deep Dive Coding Data Science Bootcamp Sample Project**

---
Robert Citek - October 5th, 2026



## Problem Definition


State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.


## Data Collection/Sources


### Data Overview


This data comes from [Kaggle](https://www.kaggle.com/c/titanic/data), which was then copied to AWS for easy access.

https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv



### Imports

```python
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn import datasets, metrics, model_selection
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
```

### File Paths

Create a variable `url` that contains the web link to the CSV data

```python
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv'    # insert url here
url
```

### Load data

Load the data into a data frame named `titanic_df`.

```python
# load data
titanic_df = pd.read_csv( url )
titanic_df
```


### head


Look at the first few rows

```python
# first few rows
titanic_df.head()
```


### tail


Look at the last few rows


```python
# last few rows
titanic_df.tail()
```


### shape


```python
# shape of data frame
titanic_df.shape
```

### size


```python
# size of data frame
titanic_df.size
```


```python
titanic_df.size?
```

```python
np.prod(titanic_df.shape)
```

### Summary meta-data


Run the `info()` method on the data frame.


```python
# display meta-data
titanic_df.info()
```

### Summary statistics

Summary statistics give you a quick overview of the data.  Some items to spot check:
- the Target:
  - look for nulls
    - null will be culled
  - look at min, max, median, and mean
    - is it a categorical field masquerading as a number?
- the Features:
  - numerical
    - look at min, max, median, and mean
      - a min=1 and max=count suggests an ID
  - categorical
    - look at unique and freq
      - freq=1, unique=count suggests an ID
      - freq=count suggests a meaningless feature


Run the `describe()` method on the data frame.

```python
# display summary statistics
titanic_df.describe()
```


### target value_counts()



The target should be the first field that you look at.  Its data type will tell you what kind of machine learning problem this is ( classification or regression. ) Any nulls should be culled.  You can check for any skew in the data set.

```python
titanic_df.value_counts?
```

```python
# value_counts on target ; include nulls
titanic_df['Survived'].value_counts( dropna = False )
```


```python
# normalized value_counts on target ; include nulls
titanic_df['Survived'].value_counts( normalize = True, dropna = False )
```

### Nulls



```python
# look for nulls
titanic_df.isnull().sum().sort_values(ascending=False)
```

```python
# look for nulls
titanic_df.isnull().sum() * 1000
# .sort_values(ascending=False)
```


```python
# look for nulls, displaying relative frequency
null_sum = titanic_df.isnull().sum()
null_freq = null_sum / titanic_df.shape[0] * 100
null_freq.sort_values()
```

### dtypes - column data types


Which data types would be appropriate for Gaussian Naive Bayes?


```python
# display data types
titanic_df.dtypes # integers and floats, NOT objects (numerical data)
```

### Unique elements



If the total number of unique items in a field equals the number of rows, that suggests those fields are unique IDs ( e.g. names, IDs ) and should be dropped.


```python
titanic_df['PassengerId'].unique()
```

```python
# number of unique elements in a column
len( titanic_df['PassengerId'].unique() )
```
```python
# check our answer
titanic_df['PassengerId'].size
```

```python
titanic_df.shape[0]
```

```python
titanic_df.nunique(axis='rows')
```





## Data Cleaning



### Backup #1


```python
# make backup
titanic_df_bak = titanic_df.copy()
titanic_df_bak
```

```python
# restore from backup
titanic_df = titanic_df_bak
titanic_df
```


### Keep only the target and one feature


Select `Survived` for the target and `Age` for the single feature

```python
# select the target and one feature
titanic_df[['Survived', 'Age']]
```

```python
titanic_df = titanic_df[['Survived', 'Age']]
titanic_df
```




### Backup #2


```python
# make backup
data_cleaning_titanic_df = titanic_df.copy()
data_cleaning_titanic_df
```

```python
# restore from backup
titanic_df = data_cleaning_titanic_df
titanic_df
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.


```python
titanic_df.isnull().sum().sort_values(ascending=False)
```

#### Impute


Impute with the median.


```python
# Calculate the median
# titanic_df['PassengerId'].size
median_age = titanic_df['Age'].median()
median_age
```

```python
# Assign median to nulls
# median_now_nulls = titanic_df['Age'].fillna(value=median_age)
titanic_df['Age'].fillna(median_age, inplace=True)
# .median()
titanic_df
```

```python
# Verify there are no nulls
titanic_df['Age'].isnull().sum()
```

```python
# Verify median is about the same as before
titanic_df['Age'].median()
```

```python
# type(median_now_nulls)
```

## Exploratory Data Analysis



GNB has two requirements:
1. the features are normally distributed
1. the features are not correlated with each other






### normal distribution of features: histogram


```python
# Display a histogram
titanic_df['Age'].hist()
# normally distributed with a positive skew
```



### No correlation with single feature




## Processing



```python
# Split data frame into target and features
target   = titanic_df['Survived']
features = titanic_df['Age']
```

```python
# Verify features
features
```

```python
# Verify target
target
```

### Single Cross Validation


Run a single Cross Validation


#### Train Test Split

```python
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.25)
```

```python
# Verify shape of each data set from Train Test Split
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

```python
train_test_split?
```

```python
X_train
```

#### Model


```python
# Create an empty GNB model
modelGNB = GaussianNB()
```

```python
modelGNB
```

#### Fit


```python
# Train ( fit ) the model
modelGNB.fit(pd.DataFrame(X_train), y_train)
```

#### Prediction


```python
# Make predicitons using the model
y_pred = modelGNB.predict(pd.DataFrame(X_test))
y_pred
```

#### Performance



We'll meassure accuracy, which is the proportions of the number correctly predicted, which is mathematically the same as one minus the proportion incorrectly predicted.

```python
# Calculate the accuracy
pred_accuracy = metrics.accuracy_score(y_test,y_pred)
pred_accuracy
```

```python
# Display a confusion matrix
# plt.figure(figsize=(8,8))
mat = metrics.confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,
  square=True,
  annot=True,
  fmt='d',
  cbar=False,
  # xticklabels=dataset.target_names,
  # yticklabels=dataset.target_names
)
plt.xlabel('true label')
plt.ylabel('predicted label');
```

## Communication of Results


### Conclusion:

#### Summary:
x-axis is what our predictions were, and y-axis is what we actually got. 141 actually survived; 73 are false positives (modeled to survive but actually perished); 1 was predicted to perish but actually survived (false negatives); and 8 were correctly modeled to have perished.

Noteworthy Points:
- The model had a large number of false positives relative to the sample size

### Results:
**Accuracy Score: 66.8%**

**Confusion Matrix:**
- **True  Positive: 141 / 223**
- **False Positive: 73  / 223**
- **False Negative: 1   / 223**
- **True  Negative: 8   / 223**


### Future work:

Do more passes and try out different techniques to improve accuracy score of model.


