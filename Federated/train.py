import os
import json
import random
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from model import get_model

from dataset import (
    create_client_dataset,
    resolve_image_path,
    load_image,
)

from fedavg import fedavg




DATA_DIR = "/content/data"

GLOBAL_TEST_CSV = (
    "/content/drive/MyDrive/centralized/"
    "global_test.csv"
)



EXPERIMENTS = {

    "iid": [
    
            "/content/drive/MyDrive/centralized/"
            "Federated/federaded_data/iid/iid/client_1.csv",
    
            "/content/drive/MyDrive/centralized/"
            "Federated/federaded_data/iid/iid/client_2.csv",
    
            "/content/drive/MyDrive/centralized/"
            "Federated/federaded_data/iid/iid/client_3.csv",
    
            "/content/drive/MyDrive/centralized/"
            "Federated/federaded_data/iid/iid/client_4.csv",
        ],
    "non_iid": [

        "/content/drive/MyDrive/centralized/"
        "Federated/federaded_data/non_iid/non_iid/client_1.csv",

        "/content/drive/MyDrive/centralized/"
        "Federated/federaded_data/non_iid/non_iid/client_2.csv",

        "/content/drive/MyDrive/centralized/"
        "Federated/federaded_data/non_iid/non_iid/client_3.csv",

        "/content/drive/MyDrive/centralized/"
        "Federated/federaded_data/non_iid/non_iid/client_4.csv",
    ],
}



OUTPUT_DIR = (
    "/content/drive/MyDrive/centralized/"
    "outputs/federated"
)



SEED = 42

BATCH_SIZE = 32

ROUNDS = 3

LOCAL_EPOCHS = 1

LEARNING_RATE = 1e-4




def set_seed(seed):

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    tf.random.set_seed(
        seed
    )




def compile_model(model):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),

        loss=tf.keras.losses.BinaryCrossentropy(),

        metrics=[

            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.AUC(
                name="auroc"
            ),

        ],
    )




def create_federated_model():

    model, base = get_model(
        architecture="densenet121"
    )

    

    base.trainable = True

    fine_tune_from = (
        len(base.layers) - 60
    )

    for layer in base.layers[
        :fine_tune_from
    ]:

        layer.trainable = False

   

    for layer in base.layers[
        fine_tune_from:
    ]:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    compile_model(
        model
    )

    return model




def create_global_test_dataset():

    df = pd.read_csv(
        GLOBAL_TEST_CSV
    )

    print(
        f"\nGlobal test samples: {len(df)}"
    )

    print(
        "\nGlobal test distribution:"
    )

    print(
        df["class_name"].value_counts()
    )

    image_paths = []

    labels = []

    

    for _, row in df.iterrows():

        image_path = resolve_image_path(
            row["filepath"],
            DATA_DIR
        )

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                f"\nImage not found:\n"
                f"{image_path}"
            )

        image_paths.append(
            image_path
        )

        labels.append(
            int(row["label"])
        )

   

    dataset = tf.data.Dataset.from_tensor_slices(

        (
            np.array(image_paths),

            np.array(
                labels,
                dtype=np.float32
            ),
        )
    )

    def load_data(
        path,
        label
    ):

        image = load_image(
            path
        )

        return image, label

    dataset = dataset.map(
        load_data,
        num_parallel_calls=2
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        1
    )

    return dataset




def train_client(

    local_model,
    global_weights,
    client_dataset,
    client_id,
):

    print(
        "\n--------------------------------------"
    )

    print(
        f"TRAINING CLIENT {client_id}"
    )

    print(
        "--------------------------------------"
    )

    

    local_model.set_weights(
        global_weights
    )

   

    compile_model(
        local_model
    )

    

    history = local_model.fit(

        client_dataset,

        epochs=LOCAL_EPOCHS,

        verbose=1,
    )

    

    client_weights = (
        local_model.get_weights()
    )

 

    client_loss = (
        history.history["loss"][-1]
    )

    client_accuracy = (
        history.history["accuracy"][-1]
    )

    client_auroc = (
        history.history["auroc"][-1]
    )

    print(
        f"\nClient {client_id} completed"
    )

    print(
        f"Loss     : {client_loss:.4f}"
    )

    print(
        f"Accuracy : {client_accuracy:.4f}"
    )

    print(
        f"AUROC    : {client_auroc:.4f}"
    )

    return (

        client_weights,

        client_loss,

        client_accuracy,

        client_auroc,

    )


def evaluate_global_model(

    model,
    test_dataset,
):

    all_labels = []

    all_probs = []

    

    for images, labels in test_dataset:

        probabilities = model(
            images,
            training=False
        )

        probabilities = tf.reshape(
            probabilities,
            [-1]
        )

        all_probs.extend(
            probabilities.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

   

    all_labels = np.array(
        all_labels,
        dtype=np.int32
    )

    all_probs = np.array(
        all_probs,
        dtype=np.float32
    )

    

    all_predictions = (

        all_probs >= 0.5

    ).astype(
        np.int32
    )

   

    metrics = {

        "accuracy": float(

            accuracy_score(
                all_labels,
                all_predictions
            )
        ),

        "auroc": float(

            roc_auc_score(
                all_labels,
                all_probs
            )
        ),

        "f1": float(

            f1_score(
                all_labels,
                all_predictions,
                zero_division=0
            )
        ),

        "precision": float(

            precision_score(
                all_labels,
                all_predictions,
                zero_division=0
            )
        ),

        "recall": float(

            recall_score(
                all_labels,
                all_predictions,
                zero_division=0
            )
        ),

        "confusion_matrix":

            confusion_matrix(
                all_labels,
                all_predictions
            ).tolist(),

    }

    return metrics




def run_experiment(

    experiment_name,

    client_csvs,

    initial_weights,

    global_test,

):

    print(
        "\n\n"
        "================================================"
    )

    print(
        f" STARTING {experiment_name.upper()} "
        "FEDERATED TRAINING"
    )

    print(
        "================================================"
    )

 

    experiment_dir = os.path.join(

        OUTPUT_DIR,

        experiment_name

    )

    model_dir = os.path.join(

        experiment_dir,

        "models"

    )

    metrics_dir = os.path.join(

        experiment_dir,

        "metrics"

    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    os.makedirs(
        metrics_dir,
        exist_ok=True
    )



    client_datasets = []

    client_sizes = []

    for client_id, csv_path in enumerate(

        client_csvs,

        start=1

    ):

        print(
            "\n======================================"
        )

        print(
            f"LOADING CLIENT {client_id}"
        )

        print(
            "======================================"
        )

        dataset, size = create_client_dataset(

            csv_path=csv_path,

            data_dir=DATA_DIR,

            batch_size=BATCH_SIZE,

            shuffle=True,

        )

        client_datasets.append(
            dataset
        )

        client_sizes.append(
            size
        )

    # --------------------------------------------------------
    # Print sizes
    # --------------------------------------------------------

    print(
        "\nCLIENT DATASET SIZES"
    )

    for i, size in enumerate(

        client_sizes,

        start=1

    ):

        print(
            f"Client {i}: {size}"
        )

    print(
        f"\nTotal samples: "
        f"{sum(client_sizes)}"
    )

    

    global_model = (
        create_federated_model()
    )

    global_model.set_weights(
        initial_weights
    )

    global_weights = (
        global_model.get_weights()
    )

   

    local_model = (
        create_federated_model()
    )

    experiment_history = []



    for round_number in range(

        1,

        ROUNDS + 1

    ):

        print(
            "\n\n"
            "================================================"
        )

        print(
            f"{experiment_name.upper()} "
            f"ROUND {round_number}/{ROUNDS}"
        )

        print(
            "================================================"
        )

        client_weights = []

        client_losses = []

        client_accuracies = []

        client_aurocs = []

        

        for client_id, client_dataset in enumerate(

            client_datasets,

            start=1

        ):

            (

                weights,

                loss,

                accuracy,

                auroc,

            ) = train_client(

                local_model,

                global_weights,

                client_dataset,

                client_id,

            )

            client_weights.append(
                weights
            )

            client_losses.append(
                float(loss)
            )

            client_accuracies.append(
                float(accuracy)
            )

            client_aurocs.append(
                float(auroc)
            )

 

        print(
            "\nPerforming FedAvg..."
        )

        global_weights = fedavg(

            client_weights,

            client_sizes,

        )

        global_model.set_weights(
            global_weights
        )

 

        print(
            "\nEvaluating global model..."
        )

        global_metrics = (
            evaluate_global_model(

                global_model,

                global_test,

            )
        )



        print(
            "\nGLOBAL TEST RESULTS"
        )

        print(
            f"Accuracy  : "
            f"{global_metrics['accuracy']:.4f}"
        )

        print(
            f"AUROC     : "
            f"{global_metrics['auroc']:.4f}"
        )

        print(
            f"F1        : "
            f"{global_metrics['f1']:.4f}"
        )

        print(
            f"Precision : "
            f"{global_metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{global_metrics['recall']:.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            np.array(
                global_metrics[
                    "confusion_matrix"
                ]
            )
        )

      

        round_model_path = os.path.join(

            model_dir,

            f"{experiment_name}_round_"
            f"{round_number:02d}.keras"

        )

        global_model.save(
            round_model_path
        )

        print(
            f"\nSaved round model:"
            f"\n{round_model_path}"
        )



        round_data = {

            "round": round_number,

            "client_sizes": client_sizes,

            "client_loss":
                client_losses,

            "client_accuracy":
                client_accuracies,

            "client_auroc":
                client_aurocs,

            "global_test":
                global_metrics,

        }

        experiment_history.append(
            round_data
        )

        history_path = os.path.join(

            metrics_dir,

            f"{experiment_name}_history.json"

        )

        with open(

            history_path,

            "w"

        ) as f:

            json.dump(

                experiment_history,

                f,

                indent=4,

            )

        print(
            f"Saved history:"
            f"\n{history_path}"
        )

   

        gc.collect()

  

    final_model_path = os.path.join(

        model_dir,

        f"{experiment_name}_fedavg_final.keras"

    )

    global_model.save(
        final_model_path
    )

    print(
        "\n"
        "================================================"
    )

    print(
        f"{experiment_name.upper()} "
        "TRAINING COMPLETE"
    )

    print(
        "================================================"
    )

    print(
        f"Final model:\n"
        f"{final_model_path}"
    )

   

    del local_model

    del global_model

    del client_datasets

    tf.keras.backend.clear_session()

    gc.collect()

    return experiment_history




def main():

    set_seed(
        SEED
    )

    print(
        "\n======================================"
    )

    print(
        " FEDERATED LEARNING"
    )

    print(
        " IID + NON-IID EXPERIMENTS"
    )

    print(
        "======================================"
    )


    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        f"\nGPUs available: {len(gpus)}"
    )

    if len(gpus) > 0:

        for gpu in gpus:

            print(
                f"GPU: {gpu}"
            )

    else:

        print(
            "WARNING: No GPU detected."
        )

    

    print(
        "\nCreating global test dataset..."
    )

    global_test = (
        create_global_test_dataset()
    )



    print(
        "\nCreating common initial model..."
    )

    initial_model = (
        create_federated_model()
    )

    initial_weights = [

        np.copy(weight)

        for weight in

        initial_model.get_weights()

    ]

 

    del initial_model

    tf.keras.backend.clear_session()

    gc.collect()

    # ========================================================
    # EXPERIMENT 1: IID
    # ========================================================
    run_experiment(

        experiment_name="iid",

        client_csvs=EXPERIMENTS["iid"],

        initial_weights=initial_weights,

        global_test=global_test,

    )


    # ========================================================
    # EXPERIMENT 2: NON-IID
    # ========================================================

    run_experiment(

        experiment_name="non_iid",

        client_csvs=EXPERIMENTS["non_iid"],

        initial_weights=initial_weights,

        global_test=global_test,

    )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    del global_test

    del initial_weights

    tf.keras.backend.clear_session()

    gc.collect()

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n\n"
        "================================================"
    )

    print(
        " ALL FEDERATED EXPERIMENTS COMPLETE"
    )

    print(
        "================================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()