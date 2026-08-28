import os
import numpy as np
import pandas as pd
import tensorflow as tf




IMG_SIZE = (224, 224)
BATCH_SIZE = 32




def resolve_image_path(filepath, data_dir="/content/data"):
    """
    Convert filepath stored in CSV into the actual local image path.
    """

    filepath = str(filepath).replace("\\", "/")



    if "/versions/1/" in filepath:

        relative_path = filepath.split(
            "/versions/1/",
            1
        )[1]

        return os.path.join(
            data_dir,
            relative_path
        )

  

    if filepath.startswith("train/"):
        return os.path.join(
            data_dir,
            filepath
        )

    if filepath.startswith("test/"):
        return os.path.join(
            data_dir,
            filepath
        )

    if filepath.startswith("val/"):
        return os.path.join(
            data_dir,
            filepath
        )

    

    if filepath.startswith(data_dir):
        return filepath

    raise ValueError(
        f"Could not resolve image path:\n{filepath}"
    )




def load_image(path):
    """
    Load and resize one X-ray image.

    DenseNet preprocessing is performed inside model.py.
    """

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image




def create_augmentation():

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal"
            ),

            tf.keras.layers.RandomRotation(
                0.04
            ),

            tf.keras.layers.RandomZoom(
                height_factor=(-0.08, 0.08),
                width_factor=(-0.08, 0.08)
            ),

            tf.keras.layers.RandomContrast(
                0.10
            ),
        ],
        name="xray_augmentation"
    )



def create_client_dataset(
    csv_path,
    data_dir="/content/data",
    batch_size=BATCH_SIZE,
    shuffle=True,
):

    

    df = pd.read_csv(
        csv_path
    )

    print(
        f"\nLoading client CSV: {csv_path}"
    )

    print(
        f"Number of samples: {len(df)}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["class_name"].value_counts()
    )

    

    image_paths = []

    for filepath in df["filepath"]:

        path = resolve_image_path(
            filepath,
            data_dir
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Image not found:\n{path}"
            )

        image_paths.append(
            path
        )


    labels = df["label"].astype(
        np.float32
    ).values

    image_paths = np.array(
        image_paths
    )

    

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels
        )
    )

    

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(df),
            reshuffle_each_iteration=True
        )



    augmentation = create_augmentation()

    def load_data(path, label):

        image = load_image(
            path
        )

        image = augmentation(
            image,
            training=True
        )

        return image, label

    
    dataset = dataset.map(
        load_data,
        num_parallel_calls=2
    )


    dataset = dataset.batch(
        batch_size
    )

    

    dataset = dataset.prefetch(
        1
    )

    return dataset, len(df)