import os
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

from utils import save_metrics



DATA_DIR = "/content/data"

GLOBAL_TEST_CSV = "/content/drive/MyDrive/centralized/global_test.csv"

MODEL_PATH = (
    "/content/drive/MyDrive/centralized/outputs/models/"
    "centralized_baseline.keras"
)

METRICS_SAVE_PATH = (
    "/content/drive/MyDrive/centralized/outputs/metrics/"
    "centralized_global_test_metrics.json"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

THRESHOLD = 0.5



def load_image(path):
    """
    Loads one X-ray image and resizes it to 224x224.

    DenseNet preprocessing is already performed inside model.py,
    so we DO NOT preprocess the image here.
    """

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image




def get_image_path(filepath):
    """
    Converts the filepath stored in global_test.csv
    into the corresponding local /content/data path.
    """

    filepath = str(filepath).replace("\\", "/")

 
    if "/versions/1/" in filepath:

        relative_path = filepath.split(
            "/versions/1/",
            1
        )[1]

        return os.path.join(
            DATA_DIR,
            relative_path,
        )

 


    if filepath.startswith("train/"):
        return os.path.join(
            DATA_DIR,
            filepath,
        )

 

    if filepath.startswith("val/"):
        return os.path.join(
            DATA_DIR,
            filepath,
        )

 

    if filepath.startswith(DATA_DIR):
        return filepath

    raise ValueError(
        f"Could not resolve image path:\n{filepath}"
    )




def create_global_test_dataset(df):

    image_paths = []
    labels = []

    for _, row in df.iterrows():

        image_path = get_image_path(
            row["filepath"]
        )

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image_paths.append(
            image_path
        )

        labels.append(
            int(row["label"])
        )

    image_paths = np.array(
        image_paths
    )

    labels = np.array(
        labels,
        dtype=np.float32,
    )

    print(
        f"\nFound {len(image_paths)} images."
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels,
        )
    )

    def load_data(path, label):

        image = load_image(path)

        return image, label

    dataset = dataset.map(
        load_data,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset




def main():

    print("\n======================================")
    print(" CENTRALIZED GLOBAL TEST EVALUATION")
    print("======================================")

   

    print(
        f"\nLoading:\n{GLOBAL_TEST_CSV}"
    )

    df = pd.read_csv(
        GLOBAL_TEST_CSV
    )

    print(
        f"Total CSV samples: {len(df)}"
    )

    print(
        "\nColumns:"
    )

    print(
        df.columns.tolist()
    )

 

    print(
        "\nClass distribution:"
    )

    print(
        df["class_name"].value_counts()
    )



    test_ds = create_global_test_dataset(
        df
    )


    print(
        "\nLoading model:"
    )

    print(
        MODEL_PATH
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

  

    print(
        "\nGenerating predictions..."
    )

    all_labels = []
    all_probs = []

    for images, labels in test_ds:

        probabilities = model.predict(
            images,
            verbose=0,
        ).reshape(-1)

        all_probs.extend(
            probabilities
        )

        all_labels.extend(
            labels.numpy()
        )

    all_labels = np.array(
        all_labels,
        dtype=np.int32,
    )

    all_probs = np.array(
        all_probs,
        dtype=np.float32,
    )

  

    all_predictions = (
        all_probs >= THRESHOLD
    ).astype(np.int32)

  

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    auroc = roc_auc_score(
        all_labels,
        all_probs,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

  

    print("\n======================================")
    print(" GLOBAL TEST RESULTS")
    print("======================================")

    print(
        f"Number of test samples : {len(all_labels)}"
    )

    print(
        f"Accuracy               : {accuracy:.4f}"
    )

    print(
        f"AUROC                  : {auroc:.4f}"
    )

    print(
        f"F1 Score               : {f1:.4f}"
    )

    print(
        f"Precision              : {precision:.4f}"
    )

    print(
        f"Recall                 : {recall:.4f}"
    )

    print(
        f"Threshold              : {THRESHOLD}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)



    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist(),
        "threshold": THRESHOLD,
        "classes": [
            "NORMAL",
            "PNEUMONIA",
        ],
        "num_test_samples": int(
            len(all_labels)
        ),
    }

    os.makedirs(
        os.path.dirname(
            METRICS_SAVE_PATH
        ),
        exist_ok=True,
    )

    save_metrics(
        metrics,
        METRICS_SAVE_PATH,
    )

    print(
        f"\nMetrics saved to:"
    )

    print(
        METRICS_SAVE_PATH
    )



if __name__ == "__main__":
    main()