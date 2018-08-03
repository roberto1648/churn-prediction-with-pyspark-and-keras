
# Churn Prediction with Pyspark and Keras

# Introduction

This work was motivated by the lack (as of August of 2018) of a distributed framework allowing modeling with arbitrary keras models. Of particular interest are multi-input neural net models that allow the use of embedding layers. During training, embedding layers allow transforming categorical variables into a meaninful vector space from which insights could be extracted. 

In order to allow the distributed training of arbitrary keras models the associated modules were developed for a simple problem. This project was seeded by [this](https://github.com/bensadeghi/pyspark-churn-prediction) very didactic github repo.

# Distributed data exploration with pyspark

The dfhelper module has a number of useful functions for exploring the data. The general approach is to:
- get global statistics going through the data using spark native functions.
- plot using pandas by extracting a sample small enough to fit in memory.


```python
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```


```python
import dfhelper

spark = dfhelper.init_session("local", "churn pred")
```


```python
dfhelper.open_spark_browser_gui()
```


```python
dfhelper.full_data_size_estimation(df_train)
```

    started full dataframe count at 2018-07-29 11:28:29.640327
    total number of rows: 2666
    ended count at 2018-07-29 11:28:29.738698
    time elapsed: 0.0980780124664 seconds
    loaded to memory the full data would occupy: 0.02644672 MB





    26446.72



This should give an idea of how big the in-memory samples or batches can be. It also gives an idea of how long the system takes to do a data-wide calculation by going once through the whole dataset to count it.

Other than that, this particular dataset is not precisely "big data" but it's intended as an example. No function requires loading the whole data into memory and thus can be used for treating actual large datasets.


```python
df_train = dfhelper.load_dataframe('/data/churn-bigml-80.csv', spark)
df_test = dfhelper.load_dataframe('/data/churn-bigml-20.csv', spark)

df_train.printSchema()
```

    root
     |-- State: string (nullable = true)
     |-- Account length: integer (nullable = true)
     |-- Area code: integer (nullable = true)
     |-- International plan: string (nullable = true)
     |-- Voice mail plan: string (nullable = true)
     |-- Number vmail messages: integer (nullable = true)
     |-- Total day minutes: double (nullable = true)
     |-- Total day calls: integer (nullable = true)
     |-- Total day charge: double (nullable = true)
     |-- Total eve minutes: double (nullable = true)
     |-- Total eve calls: integer (nullable = true)
     |-- Total eve charge: double (nullable = true)
     |-- Total night minutes: double (nullable = true)
     |-- Total night calls: integer (nullable = true)
     |-- Total night charge: double (nullable = true)
     |-- Total intl minutes: double (nullable = true)
     |-- Total intl calls: integer (nullable = true)
     |-- Total intl charge: double (nullable = true)
     |-- Customer service calls: integer (nullable = true)
     |-- Churn: boolean (nullable = true)
    



```python
dfhelper.head(df_train)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>No</td>
      <td>Yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>No</td>
      <td>Yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfhelper.summary(df_train)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>summary</th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>None</td>
      <td>100.62040510127532</td>
      <td>437.43885971492875</td>
      <td>None</td>
      <td>None</td>
      <td>8.021755438859715</td>
      <td>179.48162040510135</td>
      <td>100.31020255063765</td>
      <td>30.512404351087813</td>
      <td>200.38615903976006</td>
      <td>100.02363090772693</td>
      <td>17.033072018004518</td>
      <td>201.16894223555968</td>
      <td>100.10615153788447</td>
      <td>9.052689422355604</td>
      <td>10.23702175543886</td>
      <td>4.467366841710428</td>
      <td>2.764489872468112</td>
      <td>1.5626406601650413</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>None</td>
      <td>39.56397365334985</td>
      <td>42.521018019427174</td>
      <td>None</td>
      <td>None</td>
      <td>13.61227701829193</td>
      <td>54.21035022086982</td>
      <td>19.988162186059512</td>
      <td>9.215732907163497</td>
      <td>50.95151511764598</td>
      <td>20.16144511531889</td>
      <td>4.330864176799864</td>
      <td>50.780323368725206</td>
      <td>19.418458551101697</td>
      <td>2.2851195129157564</td>
      <td>2.7883485770512566</td>
      <td>2.4561949030129466</td>
      <td>0.7528120531228477</td>
      <td>1.3112357589949093</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>AK</td>
      <td>1</td>
      <td>408</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>43.7</td>
      <td>33</td>
      <td>1.97</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25%</td>
      <td>None</td>
      <td>73</td>
      <td>408</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>143.4</td>
      <td>87</td>
      <td>24.38</td>
      <td>165.3</td>
      <td>87</td>
      <td>14.05</td>
      <td>166.9</td>
      <td>87</td>
      <td>7.51</td>
      <td>8.5</td>
      <td>3</td>
      <td>2.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50%</td>
      <td>None</td>
      <td>100</td>
      <td>415</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>179.9</td>
      <td>101</td>
      <td>30.58</td>
      <td>200.9</td>
      <td>100</td>
      <td>17.08</td>
      <td>201.1</td>
      <td>100</td>
      <td>9.05</td>
      <td>10.2</td>
      <td>4</td>
      <td>2.75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>75%</td>
      <td>None</td>
      <td>127</td>
      <td>510</td>
      <td>None</td>
      <td>None</td>
      <td>19</td>
      <td>215.9</td>
      <td>114</td>
      <td>36.7</td>
      <td>235.1</td>
      <td>114</td>
      <td>19.98</td>
      <td>236.5</td>
      <td>113</td>
      <td>10.64</td>
      <td>12.1</td>
      <td>6</td>
      <td>3.27</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>max</td>
      <td>WY</td>
      <td>243</td>
      <td>510</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>50</td>
      <td>350.8</td>
      <td>160</td>
      <td>59.64</td>
      <td>363.7</td>
      <td>170</td>
      <td>30.91</td>
      <td>395.0</td>
      <td>166</td>
      <td>17.77</td>
      <td>20.0</td>
      <td>20</td>
      <td>5.4</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfhelper.describe(df_train)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>summary</th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
      <td>2666</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>None</td>
      <td>100.62040510127532</td>
      <td>437.43885971492875</td>
      <td>None</td>
      <td>None</td>
      <td>8.021755438859715</td>
      <td>179.48162040510135</td>
      <td>100.31020255063765</td>
      <td>30.512404351087813</td>
      <td>200.38615903976006</td>
      <td>100.02363090772693</td>
      <td>17.033072018004518</td>
      <td>201.16894223555968</td>
      <td>100.10615153788447</td>
      <td>9.052689422355604</td>
      <td>10.23702175543886</td>
      <td>4.467366841710428</td>
      <td>2.764489872468112</td>
      <td>1.5626406601650413</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>None</td>
      <td>39.56397365334985</td>
      <td>42.521018019427174</td>
      <td>None</td>
      <td>None</td>
      <td>13.61227701829193</td>
      <td>54.21035022086982</td>
      <td>19.988162186059512</td>
      <td>9.215732907163497</td>
      <td>50.95151511764598</td>
      <td>20.16144511531889</td>
      <td>4.330864176799864</td>
      <td>50.780323368725206</td>
      <td>19.418458551101697</td>
      <td>2.2851195129157564</td>
      <td>2.7883485770512566</td>
      <td>2.4561949030129466</td>
      <td>0.7528120531228477</td>
      <td>1.3112357589949093</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>AK</td>
      <td>1</td>
      <td>408</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>43.7</td>
      <td>33</td>
      <td>1.97</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max</td>
      <td>WY</td>
      <td>243</td>
      <td>510</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>50</td>
      <td>350.8</td>
      <td>160</td>
      <td>59.64</td>
      <td>363.7</td>
      <td>170</td>
      <td>30.91</td>
      <td>395.0</td>
      <td>166</td>
      <td>17.77</td>
      <td>20.0</td>
      <td>20</td>
      <td>5.4</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



## Variable distributions: histograms


```python
dfhelper.plot_histogram(df_train)
```


    <Figure size 864x864 with 0 Axes>



![png](README_files/README_15_1.png)


Most continuous variables have an approximately normal distribution. An exception is "Number of vmail messages" that seems to be bi-modal.Notice that our labels ("Churn") have two very inbalanced classes. There's a 6 to 1 ratio between the 0.0 and 1.0 classes.

## Pair-wise correlations: scatter matrix


```python
dfhelper.plot_scatter_matrix(df_train)
```


![png](README_files/README_18_0.png)


Some variables are practically 1:1 correlated with one another and then one of them should be considered.

## Correlations with the label: parallel coordinates plot


```python
dfhelper.plot_parallel_coordinates(df_train, "Churn")
```


![png](README_files/README_21_0.png)


Most variables can't separate the "Churn" value by themselves. "Total day minutes" seems to do a slightly better job along with "Total day charge".

# Data cleaning and preprocessing

Basically, some columns may be redundant because they either don't correlate with the label at all (although in that case the variable could contribute to a highly nonlinear multi-variable term, but that's usually not very likely) or some variables correlate with each other so much that one could just pick one of them (e.g., "Total day minutes" and "Total day charge"). Also, any non-numeric categorical variables have to be converted to integers or floats before attempting any machine learning. This particular dataset didn't have nans, but checking and handling them is very easy in pyspark.


```python
print "column types before processing:"
print df_train.dtypes
print

# drop some columns thought not as relevant:
drop_cols = ['State', 'Total day charge', 'Total eve charge', 
            'Total night charge', 'Total intl charge', 'Area code']
df_train_proc = dfhelper.drop_columns(df_train, drop_cols)
df_test_proc = dfhelper.drop_columns(df_test, drop_cols)

# convert string values to numeric
# can change the default string to number map when needed
df_train_proc = dfhelper.string_columns_to_numeric(df_train_proc)
df_test_proc = dfhelper.string_columns_to_numeric(df_test_proc)

# convert boolean values to numeric
df_train_proc = dfhelper.boolean_columns_to_numeric(df_train_proc)
df_test_proc = dfhelper.boolean_columns_to_numeric(df_test_proc)

print "column types after processing:"
print df_train_proc.dtypes
```

    column types before processing:
    [('State', 'string'), ('Account length', 'int'), ('Area code', 'int'), ('International plan', 'string'), ('Voice mail plan', 'string'), ('Number vmail messages', 'int'), ('Total day minutes', 'double'), ('Total day calls', 'int'), ('Total day charge', 'double'), ('Total eve minutes', 'double'), ('Total eve calls', 'int'), ('Total eve charge', 'double'), ('Total night minutes', 'double'), ('Total night calls', 'int'), ('Total night charge', 'double'), ('Total intl minutes', 'double'), ('Total intl calls', 'int'), ('Total intl charge', 'double'), ('Customer service calls', 'int'), ('Churn', 'boolean')]
    
    column types after processing:
    [('Account length', 'int'), ('International plan', 'int'), ('Voice mail plan', 'int'), ('Number vmail messages', 'int'), ('Total day minutes', 'double'), ('Total day calls', 'int'), ('Total eve minutes', 'double'), ('Total eve calls', 'int'), ('Total night minutes', 'double'), ('Total night calls', 'int'), ('Total intl minutes', 'double'), ('Total intl calls', 'int'), ('Customer service calls', 'int'), ('Churn', 'int')]



```python
spark.stop()
```

# Distributed neural net training and evaluation

## Training

The function below starts a spark session, loads and processes the training dataset, splits the data into training and validation sets, and then trains a keras neural net by drawing minibatches from the spark dataframe. This is accomplished by using batch generators during training. The class imbalance in the labels is taken into account by giving more weight to the less represented class using the "class_weight" input.

The neural net architecture consists of 3 hidden fully-connected layers (fc). The fc layers are followed by dropout layers (i.e., layers that randomly make some internal outputs equal to zero), but it was found during tests that weight regularization (i.e., adding a term to the loss such as zero weights are preferred) helped much more in training a model with better generalization (i.e., that performs better with never-seen samples).


```python
%matplotlib inline
import main; reload(main)

main. main(
    open_spark_gui=False,
    train_file='/data/churn-bigml-80.csv',
    drop_cols = ['State', 'Total day charge', 'Total eve charge',
                 'Total night charge', 'Total intl charge', 'Area code'],
    val_ratio=0.2, 
    seed=3,
    batch_size=256,
    input_shape=(None, 13),
    hidden_layers=[(256, 0.0), (256, 0.0), (256, 0.0)],
    nlabels=1,
    reg_weight=0.07,
    epochs=200,
    lr=0.001,
    decay=0,
    class_weight={0.0: 1., 1.0: 6.},
    save_to_dir='log',
    datetime_subdir=True,
)
```


![png](README_files/README_30_0.png)


    train loss: 0.435530394385
    train acc: 0.925700942378
    
    val loss: 0.470230475068
    val acc: 0.92015209794
    training history saved to: log/2018-08-02_19:12:39.536372/history.pkl
    results saved to log/2018-08-02_19:12:39.536372


## Evaluation on test set

The trained model is now evaluated on the set-aside data from the test file. Notice that the model has not seen this data either directly or indirectly during its training.


```python
reload(main)
from main import start_session, get_df, process_columns, get_scaling, train_val_split
from keras.models import load_model
import sklearn.metrics

train_file='/data/churn-bigml-80.csv'
test_file='/data/churn-bigml-20.csv'
drop_cols = ['State', 'Total day charge', 'Total eve charge',
             'Total night charge', 'Total intl charge', 'Area code']
val_ratio = 0.2
seed = 3
batch_size=256
model_path = "log/2018-08-02_19:12:39.536372/keras_model.h5"
label_col = "Churn"

def custom_acc(y, ypred):
    return sklearn.metrics.accuracy_score(y, np.round(ypred))

def custom_pres(y, ypred):
    return sklearn.metrics.precision_score(y, np.round(ypred), average=None)

def custom_recall(y, ypred):
    return sklearn.metrics.recall_score(y, np.round(ypred), average=None)

def custom_f1(y, ypred):
    return sklearn.metrics.f1_score(y, np.round(ypred), average=None)

def custom_conf_matrix(y, ypred):
    n = float(len(y))
    mat = np.array(sklearn.metrics.confusion_matrix(y, np.round(ypred))) / n
    return mat

metrics = [
    custom_acc,
    custom_pres,
    custom_conf_matrix,
    custom_recall,
    custom_f1,
]
print "Test dataset evaluation:"
print

with start_session(False) as spark:
    df_train = get_df(
        train_file, spark, False,
    )
    df_test = get_df(
        train_file, spark, False,
    )
    df_train_proc = process_columns(
        df_train, drop_cols, False,
    )
    df_test_proc = process_columns(
        df_test, drop_cols, False,
    )
    df_partial_train, __ = train_val_split(
        df_train_proc, val_ratio, seed, False,
    )
    xmeans, xstds = get_scaling(df_partial_train)
    
    model = load_model(model_path)
    xcols = [x for x in df_test_proc.columns if "Churn" not in x]
    
    scores_dict = main.distributed_scoring(
            model,
            df_test_proc,
            xcols,
            label_col,
            batch_size=32,
            total_samples=None,
            metrics=metrics,
            xscaling=(xmeans, xstds),
            yscaling=(),
    )
```

    Test dataset evaluation:
    
    evaluated metrics:
    
    custom_f1
    [0.95502162 0.73262471]
    
    custom_conf_matrix
    [[0.81285369 0.04160962]
     [0.03378286 0.11175383]]
    
    custom_acc
    0.9246075209930632
    
    custom_recall
    [0.9516209  0.77578757]
    
    custom_pres
    [0.95991178 0.73400268]
    


Considering the little amount of data available and its imbalanced nature these results are pretty good. They compare favorably with the results from a [stratified sampling analysis](https://github.com/bensadeghi/pyspark-churn-prediction). The latter considered only a balanced subset of the data which might explain the better results here in which considerably more data was used to train the model. Notice also that no grid search for optimum training parameters was conducted here.

## Conclusions

A distributed procedure for data exploration and modeling was illustrated by leveraging together the capabilities of the pyspark and keras packages. Although the data used in this illustration is relatively small, the functions developed here are capable of handling large datasets. Moreover, by illustrating a simple way for dynamically generating minibatches using pyspark, the modules here allow the modeling of structured data in multi-input keras models such as the ones that transform categorical variables to a meaninful vector space using embedding layers.
