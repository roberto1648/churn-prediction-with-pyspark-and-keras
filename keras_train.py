import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
import os
from sklearn.metrics import r2_score
from datetime import datetime
import types
import pickle
# from IPython import display
from IPython.display import clear_output

import keras
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, RMSprop
from keras import callbacks
from keras.models import load_model


# - loss -> 'binary_crossentropy' DONE (test)
# - metrics -> ['accuracy'] DONE (test)
# - inputs are now (batch_size, ncols) N/A HERE
# - change keras_generator to receive batches (will get that from pyspark), modify or delete
# "get_kth_batch"? DONE (test)
# - add live IPython display plot to training loop (code in new utils file?). what should
# then happen to "plot_training_results"? DONE (as callback) (test)

# - implement stratified sampling iterations to train model ensemble.
#  should it be handled by
# main only? maybe not check chollet's book (or web) about training on
# inbalanced datasets.
#  DONE (added "class weight" input) to force
# the program to give more weight to certain under-represented
# classes. (test)

# - optional tensorboard callback (default false). check saving
# now without tb. DONE (test)

def main(
        model=Sequential(),
        train_data=([], []), # (X, Y) or batch generator
        val_data=([], []), # (X, Y) or batch generator (no generator when using tensorboard).
        epochs=10,
        batch_size=32,
        n_train_batches=None,
        n_val_batches=None,
        loss="binary_crossentropy",
        metrics=["accuracy"], # None if not needed/wanted
        optimizer_name='rmsprop',
        lr=0.001,
        epsilon=1e-8,
        decay=0.0, # suggested: lr / epochs
        class_weight=None, # e.g., {0: 1., 1: 50., 2: 2.}
        save_to_dir='.temp_log', # empty: saves to tb_logs/current_datetime
        datetime_subdir=True,
        use_tensorboard=False,
        tensorboard_histogram_freq=10,
        ylabels=[],
        verbose=True,
):
    """
    Train the provided keras model. The train and validation data
    may be given as a (X, Y) tuples or as batch generators.
    :param model: a predefined keras model. It doesn't have to be
    compiled yet.
    :param train_data: data used for training the model.
    a tuple (X, Y) where X and Y are arrays or
     a generator that yields (Xbatch, Ybatch). The generator should
     be able to run indefinitely,
     e.g., while True: yield Xbatch, Ybatch
    :param val_data: data used for validating the model.
    a tuple (X, Y) where X and Y are arrays or
     a generator that yields (Xbatch, Ybatch). The generator should
     be able to run indefinitely,
     e.g., while True: yield Xbatch, Ybatch
    :param epochs: (int) number of iterations over the train data.
    :param batch_size: (int) number of samples in each batch.
    :param n_train_batches: (int) number of batches in the train
    generator that are processed on a single epoch. After training
    the model with n_train_batches the optimizer proceeds to the
    next epoch.
    :param n_val_batches: (int) number of batches in the validation
    generator for which the current model is evaluated.
    :param loss: (string or keras loss operator) specifies the
    scalar quantity that is directly minimized through gradient
    descent.
    :param metrics: (list of strings or keras metrics operators)
    The metrics that are evaluated after each epoch on the train
    and validation data.
    :param optimizer_name: (string or keras optimizer) specifies
    the optimization algorithm that handles how the model parameters
    are updated in view of the gradient backpropagation results in
    each epoch. It also handles updates to the learning rate which
    can thus adapt to the current stage of the optimization.
    :param lr: (float) The step by which to attempt to reduce the
    loss after each epoch. The change in loss is back propagated to
    determine how to update the model parameters.
    :param epsilon: (float) input to some optimizers.
    :param decay: (float) Input to some optimizers. Amount by which
    to reduce the learning rate after each epoch.
    :param class_weight: (dict)  Optional dictionary mapping class
    indices (integers) to a weight (float) value, used for weighting
    the loss function (during training only). This can be useful to
    tell the model to "pay more attention" to samples from an
    under-represented class.
    :param save_to_dir: (string) directory into which to save the
    various results.
    :param datetime_subdir: (boolean) Whether to create a
    subdirectory inside save_to_dir constructed from the current
    datetime string.
    :param use_tensorboard: (boolean) Whether to use the
    tensorboard callback to save internal model parameters and outputs
    along with learning rates, gradients, etc.
    :param tensorboard_histogram_freq: (int) How often to save all the
    internal variables histograms.
    :param ylabels: (list of strings) Names of the elements of the
    model output Y. Intended for plotting truth tables.
    not implemented yet.
    :param verbose: (boolean) whether to print updates during
    training.
    :return: None
    """
    optimizer = get_optimizer(
        optimizer_name, lr, epsilon, decay
    )
    compile_model(
        model, optimizer, loss, metrics
    )
    log_dir = setup_logdir(
        save_to_dir, datetime_subdir
    )
    callbacks_list = setup_callbacks(
        log_dir, use_tensorboard, tensorboard_histogram_freq,
    )
    history = fit_model(
        model, train_data, val_data, epochs,
        batch_size, n_train_batches, n_val_batches,
        class_weight,
        callbacks_list,
        False,
    )

    # if verbose:
    model_filepath = get_model_path(log_dir)
    train_score, val_score = evaluate_model(
        model_filepath, train_data, val_data,
        batch_size,
        n_train_batches, n_val_batches,
    )
        # plot_training_results()# todo

    save_history(history, log_dir)


# def get_number_of_batches(batch_size=16, npoints=0, train_data=None):
#     if npoints <= 0:
#         if type(train_data) is types.TupleType:
#             if len(train_data) == 2:
#                 X, y = train_data
#                 assert np.shape(X)[0] == np.shape(y)[0]
#                 npoints = np.shape(X)[0]
#
#     nbatches = int(np.ceil(float(npoints) / batch_size))
#
#     return nbatches


def get_optimizer(optimizer_name='rmsprop',
                 lr=0.001,
                 epsilon=1e-8,
                 decay=0.0, # suggested: lr / epochs
                  ):
    if optimizer_name == 'adagrad':
        # defaults:
        # keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        opt = Adagrad(lr=lr, epsilon=epsilon, decay=decay)
    elif optimizer_name == 'sgd':
        # defaults:
        # keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # saw on web:
        # decay_rate = initial_learning_rate / epochs
        # momentum = 0.8
        opt = SGD(lr=lr, momentum=epsilon,
                  decay=decay,
                  nesterov=False)
    elif optimizer_name == 'rmsprop':
        # defaults:
        # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = RMSprop(lr=lr, epsilon=epsilon, decay=decay)

    return opt


def compile_model(
        model, optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"], # None if not needed/wanted
):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


def setup_callbacks(
        log_dir,
        use_tensorboard=False,
        tensorboard_histogram_freq=10
):
    tb_callback = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=tensorboard_histogram_freq,
        write_graph=True,
        write_grads=True,
        write_images=True
    )
    model_filepath = get_model_path(log_dir)  # "keras_model.h5"
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_filepath,
        monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto', period=1
    )
    print "keras model (with the lowest test loss) will be saved to: {}".format(model_filepath)

    # live loss plot:
    plot_losses = PlotLosses()

    callbacks_list = [checkpoint_callback, plot_losses]

    if use_tensorboard:
        callbacks_list.append(tb_callback)
        print "tensorboard-viewable logs will be saved to folder: {}".format(log_dir)

    return callbacks_list


def fit_model(
        model, train_data, val_data, epochs,
        batch_size=None, # for train_data = tuple = (X, Y)
        n_train_batches=None, # for train_data = batch generator
        n_val_batches=None,
        class_weight=None,
        callbacks_list=[],
        verbose=False,
):
    """
    Fit a keras model by passing tuples (X, Y) or batch generators
    in place of train_data and val_data. For a batch generator
    example look at "keras_batch_generator" below.
    :param model: a keras model.
    :param train_data: (tuple (X, Y) or batch generator)
    :param val_data: (tuple (X, Y) or batch generator) has to be tuple
    if train_data is a tuple.
    :param epochs: (int) number of training iterations.
    :param batch_size: (int) number of samples per batch. Needed only
    when train_data is a tuple.
    :param nbatches:(int) number of batches per epoch. Needed only
    when train_data is a batch generator. It tells the trainer how
    many batches to draw from the generator per epoch.
    :param callbacks_list: (list of keras callback objects) callbacks
    are executed after each epoch.
    :param verbose: (bool) whether to print training progress.
    :return: history: (dict) a dictionary whose keys and values store
    the training history of the train/val loss and any metrics
    (the metrics are defined at model compilation).
    """
    if type(train_data) is types.TupleType:
        Xtrain, ytrain = train_data
        assert type(val_data) is types.TupleType
        assert batch_size is not None
        history = model.fit(
            Xtrain, ytrain,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=val_data,
            callbacks=callbacks_list,
            class_weight=class_weight,
        )
    else:
        assert n_train_batches is not None
        history = model.fit_generator(
            train_data,
            epochs=epochs,
            steps_per_epoch=n_train_batches,
            validation_data=val_data,
            validation_steps=n_val_batches,
            callbacks=callbacks_list,
            verbose=verbose,
            class_weight=class_weight,
        )

    return history.history
# def fit_model(model, train_data, val_data, epochs, batch_size, nbatches,
#               callbacks_list, verbose, **gen_kwds):
#     Xval, yval = val_data
#
#     if type(train_data) is types.TupleType:
#         Xtrain, ytrain = train_data
#         history = model.fit(Xtrain, ytrain,
#                             epochs=epochs,
#                             batch_size=batch_size,
#                             verbose=verbose,
#                             validation_data=(Xval, yval),
#                             callbacks=callbacks_list)
#     else:
#         data_gen = keras_data_generator(data_function=train_data,
#                                         batch_size=batch_size,
#                                         nbatches=nbatches,
#                                         **gen_kwds)
#         history = model.fit_generator(data_gen,
#                             epochs=epochs,
#                             steps_per_epoch=batch_size,
#                             validation_data=(Xval, yval),
#                             callbacks=callbacks_list,
#                             verbose=verbose)
#
#     return history


def evaluate_model(
        model_file_path,
        train_data=None,
        val_data=None,
        batch_size=32,
        n_train_batches=None,
        n_val_batches=None
):
    model = load_model(model_file_path)

    if train_data is not None:
        if type(train_data) is types.TupleType:
            Xtrain, ytrain = train_data
            assert batch_size is not None
            train_score = model.evaluate(
                Xtrain, ytrain, batch_size=batch_size
            )
        else:
            assert n_train_batches is not None
            train_score = model.evaluate_generator(
                train_data, steps=n_train_batches
            )
        # if metrics are included the score is a list of scalars
        # otherwise its just a scalar representing the loss
        if type(train_score) is types.ListType:
            for name, value in zip(model.metrics_names, train_score):
                print "train {}: {}".format(name, value)
        else:
            print "train loss: {}".format(train_score)

        print
    else:
        train_score = []

    if val_data is not None:
        if type(val_data) is types.TupleType:
            assert batch_size is not None
            Xval, yval = val_data
            val_score = model.evaluate(
                Xval, yval, batch_size=batch_size,
            )
        else:
            assert n_val_batches is not None
            val_score = model.evaluate_generator(
                val_data, steps=n_val_batches,
            )
        # if metrics are included the score is a list of scalars
        # otherwise its just a scalar representing the loss
        if type(val_score) is types.ListType:
            for name, value in zip(model.metrics_names, val_score):
                print "val {}: {}".format(name, value)
        else:
            print "val loss: {}".format(train_score)
    else:
        val_score = []


    # if type(train_score) is types.ListType:
    #     for name, value in zip(model.metrics_names, train_score):
    #         print "train {}: {}".format(name, value)
    # else:
    #     print "train loss: {}".format(train_score)
    #
    # print
    #
    # if type(val_score) is types.ListType:
    #     for name, value in zip(model.metrics_names, val_score):
    #         print "val {}: {}".format(name, value)
    # else:
    #     print "val loss: {}".format(train_score)

    return train_score, val_score

    # plot_training_results(model, history, Xval, yval, ylabels, "training plots")
    #
    #     print
    #     print "train loss: {}".format(train_loss)
    #     print "val loss: {}".format(val_loss)
    #     print "train score: {}".format(train_score)
    #     print "val score: {}".format(val_score)


# def evaluate_model(model_file_path, train_data, val_data, nbatches,
#                    history, ylabels, verbose):
#     model = load_model(model_file_path)
#     if verbose:
#         Xval, yval = val_data
#         val_loss = model.evaluate(Xval, yval)
#         val_score = r2_score(yval, model.predict(Xval))
#
#         if type(train_data) is types.TupleType:
#             Xtrain, ytrain = train_data
#             train_loss = model.evaluate(Xtrain, ytrain)
#             train_score = r2_score(ytrain, model.predict(Xtrain))
#         else:
#             train_loss = 0
#             train_score = 0
#
#             for batch_number in range(nbatches):
#                 Xbatch, ybatch = next(train_data)
#                 train_loss += model.evaluate(Xbatch, ybatch)
#                 train_score += r2_score(ybatch, model.predict(Xbatch))
#
#         plot_training_results(model, history, Xval, yval, ylabels, "training plots")
#
#         print
#         print "train loss: {}".format(train_loss)
#         print "val loss: {}".format(val_loss)
#         print "train score: {}".format(train_score)
#         print "val score: {}".format(val_score)


def save_history(history, log_dir):
    file_path = os.path.join(log_dir, 'history.pkl')
    with open(file_path, 'w') as fp:
        pickle.dump(history, fp)
        print "training history saved to: {}".format(file_path)
        print "results saved to {}".format(log_dir)


def setup_logdir(save_to_dir=".temp_log",
                 datetime_subdir=True):
    folder = save_to_dir
    if (save_to_dir != ".temp_log") and datetime_subdir:
        dtstr = str(datetime.now()).replace(" ", "_")
        folder = os.path.join(save_to_dir, dtstr)

    if not os.path.isdir(folder): os.makedirs(folder)

    return folder


def get_model_path(log_dir):
    return os.path.join(log_dir, "keras_model.h5")


# def keras_batch_generator(data_function=object, # function(sample_index, **kwargs) returning (x, y)
#                           batch_size=64,
#                           nbatches=20,
#                           **kwargs):
#     while True:
#         for k in range(nbatches):
#             Xbatch, ybatch = get_kth_batch(k, batch_size,
#                                            data_function, **kwargs)
#             yield Xbatch, ybatch
#
#
# def get_kth_batch(k, batch_size, data_function, **kwargs):
#     Xbatch, ybatch = [], []
#
#     for j in range(batch_size):
#         sample_index = k * batch_size + j
#         x, y = data_function(sample_index, **kwargs)
#         Xbatch.append(x)
#         ybatch.append(y)
#
#     return np.array(Xbatch), np.array(ybatch)
#


# def get_datetime_logdir():
#     dt = datetime.now()
#     log_dir = "tb_logs/{}".format(dt.strftime("%Y-%m-%d_%H:%M:%S"))
#     return log_dir


# todo:
# - divide into: plot_scores and plot_truth_tables
# - plot_scores should adapt to plot any metrics in history
# - plot_truth_tables should be concious of potential big files
# so it should have a max_samples inputs to limit the number of
# points plotted.
def plot_training_results(
        model, history, Xtest, ytest,
        ylabels=[],
        figname='training results'
):
    # plot the learning curves:
    plt.figure(str(figname) + 'loss')
    plt.semilogy(history['loss'], label='train loss')
    plt.semilogy(history['val_loss'], label='test loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])

    # predict and do a truth plot separately for each signal
    ypred = model.predict(Xtest)
    nyrows, nycols = np.shape(ypred)

    if ylabels == []:
        ylabels = ["signal {}".format(x) for x in range(nycols)]
    plt.figure(str(figname) + 'learning_curves',
               figsize=(nycols * 5, nycols * 5))
    k = 1

    # reshape ytrain or yval if needed:
    if len(ypred.shape) == 1: ypred = ypred[:, np.newaxis]
    if len(ytest.shape) == 1: ytest = ytest[:, np.newaxis]

    for yp, yt, lbl in zip(ypred.T, ytest.T, ylabels):
        ncols = nrows = int(np.ceil(np.sqrt(nycols)))
        plt.subplot(nrows, ncols, k)
        plt.plot(yt, yp, '.')
        ax = plt.gca()
        ax.annotate(lbl, xy=(0.1, 0.9), xycoords='axes fraction')
        plt.ylabel('predicted')
        plt.xlabel('test data')
        k += 1

    plt.tight_layout()
    plt.show()


# def live_loss_plot(
#         train_loss_hist=None,
#         val_loss_hist=None,
#         figname="loss history"
# ):
#     if (train_loss_hist is not None) or (val_loss_hist is not None):
#         plt.figure(figname)
#         plt.gca().cla()
#
#         if train_loss_hist is not None:
#             plt.plot(train_loss_hist, label="train")
#             # plt.semilogy(losses)
#
#         if train_loss_hist is not None:
#             plt.plot(val_loss_hist, label="val")
#
#         plt.xlabel("epoch")
#         plt.ylabel("loss")
#         plt.legend(loc="best")
#         display.clear_output(wait=True)
#         display.display(plt.gcf())


class PlotLosses(keras.callbacks.Callback):
    """
    callback for updating a loss plot after each epoch.
    """
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.semilogy(self.x, self.losses, label="loss")
        plt.semilogy(self.x, self.val_losses, label="val_loss")
        # plt.plot(self.x, self.losses, label="loss")
        # plt.plot(self.x, self.val_losses, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()


# def check_conv1d_model_inputs(X_train, X_test, y_train, y_test):
#     # add a new axis to all for the conv layers if needed:
#     if len(np.shape(X_train)) < 3:
#         Xtrain = np.copy(X_train)[:, :, np.newaxis]
#     else:
#         Xtrain = np.copy(X_train)
#
#     if len(np.shape(X_test)) < 3:
#         Xtest = np.copy(X_test)[:, :, np.newaxis]
#     else:
#         Xtest = np.copy(X_test)
#
#     if len(np.shape(y_train)) < 2:
#         ytrain = np.copy(y_train)[:, np.newaxis]
#     else:
#         ytrain = np.copy(y_train)
#
#     if len(np.shape(y_test)) < 2:
#         ytest = np.copy(y_test)[:, np.newaxis]
#     else:
#         ytest = np.copy(y_test)
#
#     return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    main()