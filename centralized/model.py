import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121


IMG_SIZE = (224, 224)


def get_model(
    architecture="densenet121"
):

    if architecture.lower() != "densenet121":

        raise ValueError(
            "This version uses DenseNet121."
        )

    inputs = layers.Input(
        shape=(*IMG_SIZE, 3),
        name="image"
    )

 

    x = tf.keras.applications.densenet.preprocess_input(
        inputs
    )


    base = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )

    base.trainable = False

   

    x = base.output

    x = layers.Dropout(
        0.25
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(
            5e-5
        ),
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.25
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_probability"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="DenseNet121_ChestXray"
    )

    return model, base