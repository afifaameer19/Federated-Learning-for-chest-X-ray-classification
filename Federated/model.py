import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121


IMG_SIZE = (224, 224)


def get_model(architecture="densenet121"):
    """
    DenseNet121 transfer-learning model for binary chest X-ray classification.

    The base is pretrained on ImageNet. DenseNet's preprocessing is applied
    inside the model so the dataset loader can continue returning float images.
    """
    if architecture.lower() != "densenet121":
        raise ValueError("This improved version uses architecture='densenet121'.")

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="image")

    
    x = tf.keras.applications.densenet.preprocess_input(inputs)

    base = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    base.trainable = False

    x = base.output
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)

    model = Model(inputs, outputs, name="DenseNet121_ChestXray")
    return model, base