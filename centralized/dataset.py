import tensorflow as tf


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def get_datasets(
    data_dir,
    batch_size=BATCH_SIZE,
    val_split=0.15,
    seed=SEED,
    img_size=IMG_SIZE,
):

    train_dir = f"{data_dir}/train"
    test_dir = f"{data_dir}/test"


    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
    )


    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )



    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names

    print("\nClasses:", class_names)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),

            tf.keras.layers.RandomRotation(
                0.03
            ),

            tf.keras.layers.RandomZoom(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05)
            ),

            tf.keras.layers.RandomContrast(
                0.08
            ),
        ],
        name="xray_augmentation",
    )

    def augment(x, y):

        x = tf.cast(
            x,
            tf.float32
        )

        x = augmentation(
            x,
            training=True
        )

        return x, y

    train_ds = train_ds.map(
        augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.cache().prefetch(
        tf.data.AUTOTUNE
    )

    test_ds = test_ds.cache().prefetch(
        tf.data.AUTOTUNE
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        class_names
    )