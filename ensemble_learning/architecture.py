from tensorflow.keras.layers import Layer
from tensorflow.nn import pool
from tensorflow import expand_dims, squeeze
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Reshape, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.layers import Reshape, Dense


class Pooling1D(Layer):
    def __init__(self, pool_size=2, pool_type="MAX", **kwargs):
        super(Pooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_type = pool_type

    def call(self, inputs):
        inputs = expand_dims(inputs, axis=-1)  # Ajouter une dimension pour le canal
        if self.pool_type == "MAX":
            output = pool(
                inputs,
                window_shape=(self.pool_size,),
                pooling_type="MAX",
                padding="VALID",
            )
        elif self.pool_type == "MIN":
            output = pool(
                -inputs,
                window_shape=(self.pool_size,),
                pooling_type="MAX",
                padding="VALID",
            )
            output = -output
        else:
            raise ValueError("pool_type must be MAX or MIN")
        output = squeeze(
            output, axis=-1
        )  # Supprimer la dimension ajoutée pour le canal
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1] // self.pool_size)
        return output_shape


def arch1(lr, input_dim=(18, 1), num_classes=1):
    model = Sequential(name="arch1")
    model.add(InputLayer(input_shape=input_dim))
    model.add(Reshape((18,)))
    model.add(
        Dense(
            num_classes,
            activation="softmax",
            kernel_constraint=non_neg(),
            bias_constraint=non_neg(),
        )
    )
    opt = Adam(learning_rate=lr)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
    )

    return model


def arch2(lr, input_dim=(18, 1), num_classes=1):
    model = Sequential(name="arch2")
    model.add(InputLayer(input_shape=input_dim))
    model.add(Reshape((18,)))
    model.add(
        Dense(
            9, activation="relu", kernel_constraint=non_neg(), bias_constraint=non_neg()
        )
    )
    model.add(Pooling1D(pool_size=3, pool_type="MIN"))
    model.add(
        Dense(
            num_classes,
            activation="softmax",
            kernel_constraint=non_neg(),
            bias_constraint=non_neg(),
        )
    )
    opt = Adam(learning_rate=lr)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
    )
    return model


def arch3(lr, input_dim=(18, 1), num_classes=1):
    model = Sequential(name="arch3")
    model.add(InputLayer(input_shape=input_dim))
    model.add(Reshape((18,)))
    model.add(
        Dense(
            9,
            activation="linear",
            kernel_constraint=non_neg(),
            bias_constraint=non_neg(),
        )
    )
    model.add(Pooling1D(pool_size=3, pool_type="MAX"))
    model.add(
        Dense(
            num_classes,
            activation="softmax",
            kernel_constraint=non_neg(),
            bias_constraint=non_neg(),
        )
    )
    opt = Adam(learning_rate=lr)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
    )
    return model


def arch4(lr, input_dim=(18, 1), num_classes=1):
    model = Sequential(name="arch4")
    model.add(InputLayer(input_shape=input_dim))
    model.add(Reshape((18,)))
    model.add(
        Dense(
            9,
            activation="linear",
            kernel_constraint=non_neg(),
            bias_constraint=non_neg(),
        )
    )
    model.add(
        Dense(
            num_classes,
            activation="softmax",
            kernel_constraint=non_neg(),
            bias_constraint=non_neg(),
        )
    )
    opt = Adam(learning_rate=lr)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
    )
    return model


def arch5(lr, input_dim=(18, 1), num_classes=1):
    model = Sequential(name="arch5")
    model.add(InputLayer(input_shape=input_dim))
    model.add(Reshape((18,)))
    model.add(Pooling1D(pool_size=2, pool_type="MAX"))
    model.add(
        Dense(
            num_classes,
            activation="softmax",
            kernel_constraint=non_neg(),
            bias_constraint=non_neg(),
        )
    )
    opt = Adam(learning_rate=lr)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
    )
    return model


def arch6(lr, input_dim=(18, 1), num_classes=1):
    model = Sequential(name="arch6")
    model.add(InputLayer(input_shape=input_dim))
    model.add(Reshape((18,)))
    model.add(Pooling1D(pool_size=2, pool_type="MIN"))
    model.add(
        Dense(
            num_classes,
            activation="softmax",
            kernel_constraint=non_neg(),
            bias_constraint=non_neg(),
        )
    )
    opt = Adam(learning_rate=lr)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
    )
    return model
