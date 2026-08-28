"""
Improved Centralized Baseline
Chest X-Ray Pneumonia Classification

Architecture:
    DenseNet121 + ImageNet

Training:
    Stage 1 -> Frozen backbone
    Stage 2 -> Fine-tuning

Evaluation:
    Validation AUROC
    Validation loss
    Accuracy
    Precision
    Recall

IMPORTANT:
The test set is used only for final evaluation.
"""


import os

import tensorflow as tf
import kagglehub

from dataset import get_datasets
from model import get_model
from utils import set_seed, plot_curves



DATA_DIR = "/content/data"

BATCH_SIZE = 32

SEED = 42

# Stage 1
HEAD_EPOCHS = 10

# Stage 2
FINETUNE_EPOCHS = 25

HEAD_LR = 3e-4

FINETUNE_LR = 5e-6


MODEL_SAVE_PATH = (
    "outputs/models/"
    "centralized_baseline.keras"
)

PLOT_SAVE_PATH = (
    "outputs/plots/"
    "centralized_training_curves.png"
)




def configure_gpu():

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        f"GPUs available: {len(gpus)}"
    )

    if gpus:

        try:

            for gpu in gpus:

                tf.config.experimental.set_memory_growth(
                    gpu,
                    True
                )

            print(
                "GPU memory growth enabled."
            )

        except RuntimeError as e:

            print(
                f"GPU configuration warning: {e}"
            )

    return gpus



def compile_model(
    model,
    learning_rate
):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),

        loss=tf.keras.losses.BinaryCrossentropy(
            label_smoothing=0.01
        ),

        metrics=[

            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.AUC(
                name="auroc",
                curve="ROC"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )




def create_callbacks():

    checkpoint = tf.keras.callbacks.ModelCheckpoint(

        MODEL_SAVE_PATH,

        monitor="val_auroc",

        mode="max",

        save_best_only=True,

        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_auroc",

        mode="max",

        factor=0.3,

        patience=2,

        min_lr=1e-7,

        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(

        monitor="val_auroc",

        mode="max",

        patience=6,

        restore_best_weights=True,

        verbose=1,
    )

    return [
        checkpoint,
        reduce_lr,
        early_stop,
    ]




def main():

   

    set_seed(SEED)

    

    configure_gpu()

    print(
        f"\nDataset directory: {DATA_DIR}"
    )

    

    (
        train_ds,
        val_ds,
        test_ds,
        classes
    ) = get_datasets(

        data_dir=DATA_DIR,

        batch_size=BATCH_SIZE,

        seed=SEED,
    )

    print(
        f"\nClasses: {classes}"
    )

    

    print(
        "\nCreating DenseNet121 model..."
    )

    model, base = get_model(
        "densenet121"
    )

    model.summary()

    os.makedirs(
        os.path.dirname(
            MODEL_SAVE_PATH
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(
            PLOT_SAVE_PATH
        ),
        exist_ok=True
    )

   

    print(
        "\n"
        + "=" * 60
    )

    print(
        "STAGE 1: TRAINING CLASSIFICATION HEAD"
    )

    print(
        "=" * 60
    )

    base.trainable = False

    compile_model(
        model,
        HEAD_LR
    )

    history1 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=HEAD_EPOCHS,

        callbacks=create_callbacks(),

        verbose=1,
    )

    

    print(
        "\n"
        + "=" * 60
    )

    print(
        "STAGE 2: FINE-TUNING DENSENET121"
    )

    print(
        "=" * 60
    )

  

    base.trainable = True

    
    fine_tune_from = max(
        0,
        len(base.layers) - 100
    )

    for layer in base.layers:

        layer.trainable = False

    for layer in base.layers[
        fine_tune_from:
    ]:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True

    trainable_layers = sum(
        int(layer.trainable)
        for layer in model.layers
    )

    print(
        f"\nTrainable layers: "
        f"{trainable_layers}"
    )

   

    compile_model(
        model,
        FINETUNE_LR
    )

    

    history2 = model.fit(

        train_ds,

        validation_data=val_ds,

        initial_epoch=HEAD_EPOCHS,

        epochs=(
            HEAD_EPOCHS
            + FINETUNE_EPOCHS
        ),

        callbacks=create_callbacks(),

        verbose=1,
    )

    

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LOADING BEST MODEL"
    )

    print(
        "=" * 60
    )

    best_model = tf.keras.models.load_model(
        MODEL_SAVE_PATH
    )

   

    print(
        "\n========== BEST MODEL VALIDATION =========="
    )

    val_results = best_model.evaluate(
        val_ds,
        verbose=1
    )

    for name, value in zip(
        best_model.metrics_names,
        val_results
    ):

        print(
            f"{name}: {value:.4f}"
        )

  

    print(
        "\n========== FINAL TEST =========="
    )

    test_results = best_model.evaluate(
        test_ds,
        verbose=1
    )

    for name, value in zip(
        best_model.metrics_names,
        test_results
    ):

        print(
            f"{name}: {value:.4f}"
        )

    

    best_model.save(
        MODEL_SAVE_PATH
    )

    print(
        f"\nBest model saved to:"
        f"\n{MODEL_SAVE_PATH}"
    )

    

    class CombinedHistory:

        def __init__(
            self,
            h1,
            h2
        ):

            self.history = {}

            keys = (
                set(h1.history)
                |
                set(h2.history)
            )

            for key in keys:

                self.history[key] = (

                    h1.history.get(
                        key,
                        []
                    )

                    +

                    h2.history.get(
                        key,
                        []
                    )
                )

    combined_history = CombinedHistory(
        history1,
        history2
    )

    

    plot_curves(
        combined_history,
        PLOT_SAVE_PATH
    )

    print(
        "\nTraining complete."
    )


if __name__ == "__main__":

    main()