import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
from datetime import datetime
import webbrowser
from pandas.plotting import parallel_coordinates

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import DataFrame
import pyspark.sql.functions
import pyspark.sql.types


def init_session(
        master="local",
        appName="new spark app",
):
    sc = SparkContext(master=master, appName=appName)
    spark = SparkSession(sc)
    return spark


def load_dataframe(
        fname, spark_session,
        file_format='com.databricks.spark.csv', # csv format: 'com.databricks.spark.csv'
        header="true",
        inferSchema="true",
):
    if not fname.startswith("."): fname = "." + fname
    df = spark_session.read.load(
        fname,
        format=file_format,
        header=header,
        inferSchema=inferSchema,
    )
    return df


def head(df):
    return pd.DataFrame(df.take(5), columns=df.columns)


def describe(df):
    return df.describe().toPandas()


def summary(df):
    return df.summary().toPandas()


def unique(df, col_name):
    return df.select(col_name).distinct().toPandas()


def full_data_size_estimation(df):
    """
    Estimate the memory that the full data would occupy if loaded to Ram. Intended to give an
    idea of how big the sample sizes should be during data analysis. Also printed out is how
    long it takes to go trhough the whole dataset for counting its rows. This also gives an
    idea of how much would it take to do dataframe-wide operations (e.g., such as shuffling).
    :param df: spark dataframe.
    :return: (float) data size in bytes.
    """
    print "started full dataframe count at {}".format(datetime.now())
    ti = time.time()
    nrows = df.count()
    tf = time.time()
    print "total number of rows: {}".format(nrows)
    print "ended count at {}".format(datetime.now())
    print "time elapsed: {} seconds".format(tf - ti)

    if (nrows > 100):
        data_size = sys.getsizeof(df.take(100))
        data_size *= nrows / 100.
    else:
        data_size = sys.getsizeof(df.collect())

    print "loaded to memory the full data would occupy: {} MB".format(data_size / 1e6)
    return data_size


def list_numeric_features(df):
    numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'double']
    return numeric_features


def list_non_numeric_features(df):
    nonnum = [x for x in df.columns if x not in list_numeric_features(df)]
    return nonnum


def drop_column(df, col_name):
    if hasattr(df, col_name):
        df = df.drop(col_name)
    else:
        print "could not erase column"
        print 'dataframe does not have a column named "{}"'.format(col_name)

    return df


def drop_columns(df, col_names=[]):
    for col_name in col_names:
        df = drop_column(df, col_name)
    return df


def rename_column(df, old_name, new_name):
    if hasattr(df, old_name):
        df = df.withColumnRenamed(old_name, new_name)
    else:
        print "could not rename column"
        print 'dataframe does not have a column named "{}"'.format(old_name)
    return df


def rename_columns(df, old_new_names=[()]):
    for old, new in old_new_names:
        df = rename_column(df, old, new)
    return df


def modify_column(df, col_name, new_values):
    if hasattr(df, col_name):
        df = df.withColumn(col_name, new_values)
    else:
        print "could not modify column"
        print 'dataframe does not have a column named "{}"'.format(col_name)
    return df


def modify_columns(df, names_values=[()]):
    for name, values in names_values:
        df = modify_column(df, name, values)
    return df


def get_column_dtype(df, col_name):
    dtype = None

    for name, col_dtype in df.dtypes:
        if name == col_name:
            dtype = col_dtype
            break

    if dtype is None:
        print "could not find column type"
        print 'dataframe does not have a column named "{}"'.format(col_name)

    return dtype


def get_dtype_columns(df, dtype="string"):
    return [x for x, col_type in df.dtypes if col_type == dtype]


def string_column_to_numeric(
        df, col_name,
        str2num_map={
            'Yes': 1, 'No':0,
            'yes':1, 'no':0,
            'true':1, 'false':0,
        }
):
    """
    Transforms column values from strings to spark integers. Done by applying the provided map in
    str2num_map. This can then handle transforming strings to binary (0, 1) or any numeric
    representation.
    :param df: spark dataframe
    :param col_name: (string) name of the dataframe column to modify.
    :param str2num_map: (dict) string to integer values map.
    :return: spark dataframe with a col_name with numeric values.
    """
    # toNum = pyspark.sql.functions.UserDefinedFunction(
    #     lambda k: str2num_map[k], pyspark.sql.types.IntegerType()
    # )
    toNum = pyspark.sql.functions.UserDefinedFunction(
        lambda k: str2num_map[k], pyspark.sql.types.DoubleType()
    )
    col_type = get_column_dtype(df, col_name)

    if col_type == 'string':
        df = modify_column(df, col_name, toNum(df[col_name]))
    else:
        print "could not convert column from string to numeric"
        print 'column {} dtype is {}'.format(col_name, col_type)

    return df


def string_columns_to_numeric(
        df, col_names=[],
        str2num_map={
            'Yes': 1., 'No': 0.,
            'yes': 1., 'no': 0.,
            'true': 1., 'false': 0.,
        }
):
    """
    Transforms columns values from strings to spark integers. Done by applying the provided map in
    str2num_map. This can then handle transforming strings to binary (0, 1) or any numeric
    representation.
    :param df: spark dataframe
    :param col_names: (list of strings) names of the dataframes columns to modify. If not given, all
    the existing string type columns are modified.
    :param str2num_map: (dict) string to integer values map.
    :return: spark dataframe with col_names with numeric values.
    """
    if col_names == []:
        col_names = [x for x, t in df.dtypes if t == 'string']

    for col_name in col_names:
        df = string_column_to_numeric(df, col_name, str2num_map)

    return df


def boolean_column_to_numeric(df, col_name):
    """
    Transforms column values from boolean to (0, 1).
    :param df: spark dataframe
    :param col_name: (string) name of the dataframe column to modify.
    :return: spark dataframe with a col_name with numeric values.
    """
    col_type = get_column_dtype(df, col_name)

    if col_type == 'boolean':
        # df = modify_column(
        #     df, col_name, df[col_name].cast(pyspark.sql.types.IntegerType())
        # )
        df = modify_column(
            df, col_name, df[col_name].cast(pyspark.sql.types.DoubleType())
        )
    else:
        print "could not convert column from boolean to numeric"
        print 'column {} dtype is {}'.format(col_name, col_type)
    return df


def boolean_columns_to_numeric(
        df, col_names=[],
):
    """
    Transforms columns values from boolean to (0, 1).
    :param df: spark dataframe
    :param col_names: (list of strings) names of the dataframe columns to modify. If not given,
    then all the boolean type columns are modified.
    :return: spark dataframe with a col_names with numeric values.
    """
    if col_names == []:
        col_names = [x for x, t in df.dtypes if t == 'boolean']

    for col_name in col_names:
        df = boolean_column_to_numeric(df, col_name)
    return df


def plot_histogram(df, sample_size=0.1, figsize=(12, 12)):
    """
    Take a sample from df, convert to Pandas and plot a histogram of the valid individual columns.
    :param df: spark dataframe
    :param sample_size: (float) proportion of the data to sample for the plot.
    :param figsize: (tuple)
    :return: None
    """
    sampled_df = df.sample(False, sample_size).toPandas()
    plt.figure(figsize=figsize)
    sampled_df.hist(figsize=figsize)


def plot_scatter_matrix(df, sample_size=0.1, figsize=(12, 12)):
    """
    Take a sample from df, convert to Pandas and then plot a scatter matrix to see how the
    various columns correlate with one another.
    :param df: spark dataframe.
    :param sample_size: (float) proportion of the data to sample for the plot.
    :param figsize: (tuple)
    :return: None
    """
    numeric_features = list_numeric_features(df)
    sampled_data = df.select(numeric_features).sample(False, sample_size).toPandas()

    axs = pd.scatter_matrix(sampled_data, figsize=figsize);

    # Rotate axis labels and remove axis ticks
    n = len(sampled_data.columns)
    for i in range(n):
        v = axs[i, 0]
        v.yaxis.label.set_rotation(0)
        v.yaxis.label.set_ha('right')
        v.set_yticks(())
        h = axs[n - 1, i]
        h.xaxis.label.set_rotation(90)
        h.set_xticks(())


def plot_parallel_coordinates(
        df, col_name, sample_size=0.1,
        scale_cols=True,
        figsize=(20, 10),
):
    """
    Take a sample from df, convert to Pandas and do on it a parallel coordinates plot of
    col_name vs other columns. Intended to see how the features correlate with the label
    and/or whether some features can separate the label values.
    :param df: spark dataframe.
    :param col_name: (string) column indicating the y axis values for all the lines.
    :param sample_size: (float) proportion of the data to sample for the plot.
    :param scale_cols: (bool) Whether to scale the features to the [0., 1.] interval.
    :param figsize: (tuple)
    :return: None
    """
    numeric_features = list_numeric_features(df)
    if col_name not in numeric_features: numeric_features += [col_name]
    sampled_df = df.select(numeric_features).sample(False, sample_size).toPandas()

    if scale_cols:
        for name in sampled_df.columns:
            if name != col_name:
                sampled_df[name] -= sampled_df[name].min()
                sampled_df[name] /= sampled_df[name].max()

    plt.figure(figsize=figsize)
    parallel_coordinates(sampled_df, col_name)
    plt.xticks(rotation=90)


# this can be an example for keras, will need to yield Xbatch, ybatch instead though.
def minibatch_generator(df, batch_size=64):
    """
    Generator for iteratively getting the rows in the dataframe df in the form of batches. This
     is done by creating a new repartitioned dataframe and then iterating over the partitions.
     During repartitioning the full data is shuffled (which might take a while in the case of very
     large datasets). The generator goes first through all the rows on df and is able to go
     through many passes indefinitely.
    :param df: a spark dataframe
    :param batch_size: (int) number of rows in the generated batches.
    :yield: (pd.DataFrame) batches converted to pandas DataFrames.
    e.g.,
    gen = minibatch_generator(df, 64)
    batch = next(gen)
    len(batch)
    # 64
    """
    nrows = df.count() # this might be slow for a large dataframe, might want to extract it from summary if that's precalculated.
    nbatches = int(nrows) // int(batch_size)
    dfpart = df.repartition(nbatches) # this may also be slow since it requires shuffling
    # the whole dataframe, but there's no better way to do it that i know of.

    while True:
        for partition in dfpart.rdd.mapPartitions(lambda part: [list(part)]).toLocalIterator():
            yield pd.DataFrame(partition)


def open_spark_browser_gui():
    """
    Open a browser with the spark session gui. Refresh to see the latest changes. When session
    is closed will show an error.
    :return: None
    """
    webbrowser.open("http://localhost:4040")


def close_session(spark_session):
    spark_session.stop()