from keras.layers import Dense, Dropout, Flatten
from keras.models import Input, Model
import keras


def model1(
        input_shape=(None, 20),
        hidden_layers=[(4, 0.25), (4, 0.25)],
       nlabels=1,
       verbose=True,
):
    inp = x = Input(batch_shape=input_shape, name='input')

    for layer_number, (n_units, dropout_rate) in enumerate(hidden_layers):
        name = "hidden_{}".format(layer_number)
        x = Dense(n_units, activation='relu', name=name)(x)
        if dropout_rate > 0:
            name = "dropout_{}".format(layer_number)
            x = Dropout(dropout_rate, name=name)(x)

    x = Dense(nlabels, activation='linear', name='predictions')(x)
    model = Model(inputs=inp, outputs=x)

    if verbose: print model.summary()

    return model


def model2(
    input_shape=(None, 20),
    hidden_layers=[(4, 0.25), (4, 0.25)],
    nlabels=1,
    reg_weight=0.01,
    verbose=True,
):
    inp = x = Input(batch_shape=input_shape, name='input')

    for layer_number, (n_units, dropout_rate) in enumerate(hidden_layers):
        name = "hidden_{}".format(layer_number)
        x = Dense(n_units,
                  activation='relu',
                  name=name,
                  kernel_regularizer=keras.regularizers.l2(reg_weight),
                  )(x)
        if dropout_rate > 0:
            name = "dropout_{}".format(layer_number)
            x = Dropout(dropout_rate, name=name)(x)

    x = Dense(nlabels, activation='linear', name='predictions')(x)
    model = Model(inputs=inp, outputs=x)

    if verbose: print model.summary()

    return model

