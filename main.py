import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Dataset
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
size = (IMAGE_HEIGHT, IMAGE_WIDTH)
BATCH_SIZE = 32
SEED_TRAIN_VALIDATION = 1
SHUFFLE = True
VALIDATION_SPLIT = 0.2

# Paths
log_dir_fit = "logs/fit/"
log_dir_plots = "logs/plots/"
path_images = "images/"

# Data Augmentation
RANDOM_ROTATION = 0.1

# MobileNet
transfer_learning = False

# normal Training
TRAINING_EPOCHS = 20

# Fine Tuning
do_fine_tuning = True
FINE_TUNING_EPOCHS = 3


def evaluate_model(model_to_evaluate, test_dataset):
    loss, acc = model_to_evaluate.evaluate(test_dataset)
    print("model accuracy: {:5.2f}%".format(100 * acc))
    return


def generate_plot_example_images_labeled():
    image_batch, label_batch = next(iter(train_ds))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        class_identifier = "Face" if label_batch[i].numpy() == 1 else "No Face"
        plt.title(class_identifier)
        plt.axis("off")
    plt.savefig(fname=log_dir_plots + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), bbox_inches='tight')
    return


def get_callbacks():
    # log data for tensorBoard
    dir_path = log_dir_fit + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    return [
        tf.keras.callbacks.TensorBoard(log_dir=dir_path, histogram_freq=1),
        # tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=3)
    ]


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# API_DOCS https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(directory=path_images,
                                                       labels='inferred',
                                                       batch_size=BATCH_SIZE,
                                                       image_size=size,
                                                       shuffle=SHUFFLE,
                                                       seed=SEED_TRAIN_VALIDATION,
                                                       validation_split=VALIDATION_SPLIT,
                                                       subset="training")

validation_ds = tf.keras.utils.image_dataset_from_directory(directory=path_images,
                                                            labels='inferred',
                                                            batch_size=BATCH_SIZE,
                                                            image_size=size,
                                                            shuffle=SHUFFLE,
                                                            seed=SEED_TRAIN_VALIDATION,
                                                            validation_split=VALIDATION_SPLIT,
                                                            subset="validation")

# split into validate and test
val_batches = tf.data.experimental.cardinality(validation_ds)  # number of Batches
test_ds = validation_ds.take((2 * val_batches) // 3)
validation_ds = validation_ds.skip((2 * val_batches) // 3)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(RANDOM_ROTATION),
    ]
)

if transfer_learning:
    TRAIN_BASE_MODEL = False
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=(size[0], size[1], 3),
                                                        include_top=False,
                                                        weights="imagenet",
                                                        classes=2,
                                                        include_preprocessing=False,
                                                        )
else:
    TRAIN_BASE_MODEL = True
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=(size[0], size[1], 3),
                                                        include_top=False,
                                                        weights=None,
                                                        classes=2,
                                                        include_preprocessing=False,
                                                        )

# Freeze MobileNet
base_model.trainable = TRAIN_BASE_MODEL

inputs = keras.Input(shape=(size[0], size[1], 3))
x = data_augmentation(inputs)

# scale -1 to 1 for mobileNet
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# BatchNormalization inference mode set with training
x = base_model(x, training=TRAIN_BASE_MODEL)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)

# binary output, yes or no
outputs = keras.layers.Dense(units=1)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="myModel")

model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(0.0005),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

model.fit(train_ds, epochs=TRAINING_EPOCHS, validation_data=validation_ds,
          callbacks=get_callbacks())

if do_fine_tuning:
    # Fine-Tuning
    # unfreeze base_model in case it was frozen. Very low learning Rate required to not overfit MobileNet

    # Unfreeze
    base_model.trainable = True
    model.summary()

    # 1e-5 = 0.00001
    # because the model may be frozen earlier, compile is needed to make the model trainable again
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    model.fit(train_ds, epochs=FINE_TUNING_EPOCHS, validation_data=validation_ds, callbacks=get_callbacks())

evaluate_model(model, test_ds)
