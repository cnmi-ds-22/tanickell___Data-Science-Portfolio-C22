# Project 1: Transactions – Pass 00




## Problem Definition




#### 1. State the business problem.

Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.




> Response




## Data Collection




#### 2. Load Pandas, Numpy, and Matplotlib modules

```python
# imports here
```

#### 3. Load data Train.csv from AWS S3.

```python
# url here
```

- Set number of columns and rows for this pass:

```python
# set cols and rows for pass
```

- Read in the data from .csv using pandas:

```python
# make dataframe
```

- Check the dataframe:

```python
# check dataframe
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
```

- head

```python
# head
```

- tail

```python
# tail
```

- shape

```python
# shape
```

- size

```python
# size
```

- info

```python
# info
```

- describe

```python
# describe
```

- nunique

```python
# nunique
```

- unique

```python
# unique
```

- isnull

```python
# isnull
```

```python
# isnull frequencies
```

- dtypes

```python
# dtypes
```

#### 5. NOTE: the ‘target’ column indicates a successful transaction (‘1’) or a no-transaction (‘0’). Verify these are the only values in that column, and that there are no nulls.

```python
# value counts on target
```

```python
# value counts normalized
```

```python
# target col size/shape
```

```python
# sum of nulls in target
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
# head again to double-check
```

```python
# 'Unnamed' and 'ID_code are very likely identifier columns, and should be dropped

# target, var_0, and var_1 are meaningful data columns and will be kept
```

```python
# make backup
```

```python
# restore from backup
```

- This pass, keep only the target and two features:

```python
# keep only the target and two features
```

```python
# check our dataframe (display transformed df)
```

- backup 2

```python
# backup 2
```

```python
# restore from backup 2
```

```python
# No rows have nulls, so we can skip imputing with fillna() and/or dropping rows
```

## Exploratory Data Analysis

Produce some visual analysis of the data.  Recall that Gaussian Naive Bayes assumes the features are normally distributed and independent. Note: you might have to do multiple plots in groups.




#### 7. Check the distributions of all variables by plotting.

```python
# Display a histogram
```

```python
# display a boxplot
```

#### 8. Check the correlation values between all **feature columns** to ensure there are no substantial correlations between features. This is important to support the decision to classify the ‘target’ using Naïve Bayes.



```python
# column correlations
```

```python
# Display correlation matrix
```

#### 9. Create two data frames: one with all successful transactions, one with all unsuccessful transactions. **Make sure they are copies and not slices**.




```python
# make successful (copy)
```

```python
# make unsuccessful (copy)
```

## Data Processing




### Single Cross-validation ( CV )




#### 10. Create two data sets: one with all the feature columns (everything except for Unnamed: 0, ID_code and target) and one with just the target. Make sure they are copies and not slices.  And make sure they are the correct dimensions.



```python
# make target and features
```

```python
# check target
```

```python
# check target value counts
```

```python
# check target dimensions and type
```

```python
# check features
```

```python
# check features dimensions and type
```

#### 11. Define an empty Gaussian Naïve Bayes model using Sklearn.



```python
# Create an empty GNB model
```

#### 12. Divide the two data sets you created in step #10 into training and testing subsets, i.e. train-test split.



```python
# Train Test Split
```

#### 13. Train ( fit ) the model using the training subset of the dataset.



```python
# Train ( fit ) the model
```

#### 14. Test the model ( predict ) using the testing subset of the dataset.



```python
# Make predictions using the model
```

#### 15. Calculate and report the accuracy.



```python
# Calculate the accuracy and save to array
```

#### 16. Present a confusion matrix and heatmap



```python
# Display confusion matrix
```

#### 17. Present the results of a Classification Report (sklearn.metrics.classification_report).

```python
# 1. Generate the report (as a dictionary, for formatting)

# 2. Convert to DataFrame and transpose
```
