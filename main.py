import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics

import keras_models
import keras_train
import dfhelper


#todo:
"""
this is more like an orchestra director, will:
- construct a pyspark df. here goes the droppings and convertion to numeric.
- divide the train set into train_partial and validation dfs
- normalize all columns in train set. use the same scaling for val and test.
- get a set of stratified samples.
- define the model, get it from keras_models.
- setup the generator call to collect minibatches from pyspark.
- train one model per each stratified sample.
- evaluate with test on ensemble.
"""


def main(
        open_spark_gui=False,
        train_file='/data/churn-bigml-80.csv',
        drop_cols = ['State', 'Total day charge', 'Total eve charge',
                     'Total night charge', 'Total intl charge', 'Area code'],
        val_ratio=0.2,
        seed=3,
        batch_size=256,
        input_shape=(None, 13),
        hidden_layers=[(64, 0.0), (64, 0.0), (64, 0.0)],
        nlabels=1,
        reg_weight=0.07,
        epochs=50,
        lr=0.001,
        decay=0,  # 0.001 / 400, # suggested: lr / epochs
        class_weight={0.0: 1., 1.0: 6.},  # e.g., {0: 1., 1: 50., 2: 2.}
        save_to_dir='.temp_log',
        datetime_subdir=True, # not active when save_to_dir=".temp_log"
):
    with start_session(open_spark_gui) as spark:
        # spark = start_session(open_spark_gui)
        df_train = get_df(
            train_file, spark
        )
        df_train_proc = process_columns(
            df_train, drop_cols
        )
        df_partial_train, df_val = train_val_split(
            df_train_proc, val_ratio, seed,
        )
        n_train_batches, n_val_batches = get_number_of_batches(
            df_partial_train, df_val, batch_size,
        )
        xmeans, xstds = get_scaling(df_partial_train)
        train_gen = batch_generator(
            df_partial_train, n_train_batches, xmeans, xstds,
        )
        val_gen = batch_generator(
            df_val, n_val_batches, xmeans, xstds,
        )
        model = keras_models.model2(
            input_shape=input_shape,
            hidden_layers=hidden_layers,
            nlabels=nlabels,
            reg_weight=reg_weight,
            verbose=True,
        )
        keras_train.main(
            model=model,
            train_data=train_gen,  # (X, Y) or batch generator
            val_data=val_gen,  # (X, Y) or batch generator (no generator when using tensorboard).
            epochs=epochs,
            batch_size=batch_size,
            n_train_batches=n_train_batches,
            n_val_batches=n_val_batches,
            loss="binary_crossentropy",
            metrics=["accuracy"],  # None if not needed/wanted
            optimizer_name='rmsprop',
            lr=lr,
            epsilon=1e-8,
            decay=decay,  # 0.001 / 400, # suggested: lr / epochs
            class_weight=class_weight,  # e.g., {0: 1., 1: 50., 2: 2.}
            save_to_dir=save_to_dir,  # empty: saves to tb_logs/current_datetime
            datetime_subdir=datetime_subdir,
            use_tensorboard=False,
            tensorboard_histogram_freq=10,
            ylabels=[],
            verbose=False,
        )


def distributed_scoring(
        model,
        spark_df,
        xcols,
        label_col,
        batch_size=32,
        total_samples=None,
        metrics=[],
        xscaling=(),
        yscaling=(),
):
    all_cols = spark_df.columns
    if total_samples is None: total_samples = spark_df.count()
    nbatches = total_samples // batch_size

    if metrics is []: metrics = [sklearn.metrics.mean_squared_error]
    scores = []

    dfpart = spark_df.repartition(nbatches)

    for partition in dfpart.rdd.mapPartitions(lambda part: [list(part)]).toLocalIterator():
        batch_df = pd.DataFrame(partition)
        batch_df.columns = all_cols
        X = batch_df[xcols].values
        y = batch_df[label_col].values

        if xscaling is not ():
            xmeans, xstds = xscaling
            X -= xmeans
            X /= xstds

        if yscaling is not ():
            ymeans, ystds = yscaling
            y -= ymeans
            y /= ystds

        ypred = model.predict(X)
        ypred = np.reshape(ypred, np.shape(y))
        # print sklearn.metrics.accuracy_score(y, np.round(ypred))
        # print sklearn.metrics.precision_score(y, np.round(ypred), average=None)
        scores.append([np.array(metric(y, ypred)) / float(nbatches) for metric in metrics])

    # averages over the batches. Should do it one by one because
    # the metrics can have different ndarray shapes
    mean_scores = scores[0]

    for scores_row in scores[1:]:
        for k in range(len(mean_scores)):
            mean_scores[k] += scores_row[k]

    # pack the results:
    scores_dict = {}

    for metric, score in zip(metrics, mean_scores):
        scores_dict[metric.__name__] = score

    print "evaluated metrics:"
    print
    for key, value in scores_dict.iteritems():
        print key
        print value
        print

    return scores_dict


def start_session(open_spark_gui=False):
    spark = dfhelper.init_session("local", "churn pred")
    if open_spark_gui: dfhelper.open_spark_browser_gui()
    return spark


def get_df(
        train_file='/data/churn-bigml-80.csv',
        spark_session=None,
        verbose=True,
):
    df_train = dfhelper.load_dataframe(train_file, spark_session)
    if verbose: df_train.printSchema()
    return df_train


def process_columns(
        df_train,
        drop_cols=['State', 'Total day charge', 'Total eve charge',
                   'Total night charge', 'Total intl charge', 'Area code'],
        verbose=True,
):
    if verbose:
        print "column types before processing:"
        print df_train.dtypes
        print

    # drop some columns thought not as relevant:
    df_train_proc = dfhelper.drop_columns(df_train, drop_cols)

    # convert string values to numeric
    # can change the default string to number map when needed
    df_train_proc = dfhelper.string_columns_to_numeric(df_train_proc)

    # convert boolean values to numeric
    df_train_proc = dfhelper.boolean_columns_to_numeric(df_train_proc)

    if verbose:
        print "column types after processing:"
        print df_train_proc.dtypes
        print

    return df_train_proc


def train_val_split(
        df_train_proc,
        val_ratio=0.2,
        seed=3,
        verbose=True,
):
    # train-val split:
    df_partial_train, df_val = df_train_proc.randomSplit(
        [1-val_ratio, val_ratio], seed=seed
    )
    if verbose:
        print "train val total number of points: ", df_partial_train.count(), df_val.count()
        print
    return df_partial_train, df_val


def get_number_of_batches(df_partial_train, df_val, batch_size):
    # determine the number of batches
    batch_size = 256
    npoints_train = df_partial_train.count()  # ;print npoints
    n_train_batches = npoints_train // batch_size
    npoints_val = df_val.count()  # ;print npoints
    n_val_batches = npoints_val // batch_size
    print "train val nbatches: ", n_train_batches, n_val_batches
    return n_train_batches, n_val_batches


def get_scaling(df_partial_train):
    # means and stds of df_partial_train columns:
    stats = df_partial_train.describe().toPandas()
    stats = stats.set_index("summary")
    xcols = [x for x in stats.columns if "Churn" not in x]
    xmeans = stats[xcols].loc["mean"].values
    xmeans = [float(x) for x in xmeans]
    xstds = stats[xcols].loc["stddev"].values
    xstds = [float(x) for x in xstds]
    return xmeans, xstds


def batch_generator(df, nbatches, xmeans, xstds):
    all_cols = df.columns
    xcols = [x for x in all_cols if "Churn" not in x]
    dfpart = df.repartition(nbatches)

    while True:
        for partition in dfpart.rdd.mapPartitions(lambda part: [list(part)]).toLocalIterator():
            batch_df = pd.DataFrame(partition)
            batch_df.columns = all_cols
            X = batch_df[xcols].values
            y = batch_df["Churn"].values
            X -= xmeans
            X /= xstds
            yield (X, y)
