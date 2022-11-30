import tensorflow as tf

breeds = ["beagle", "bernese_mountain_dog", "doberman", "labrador_retriever", "siberian_husky"]
# Dictionary of parameters
args = {
    "labels": "inferred",
    "label_mode": "categorical",
    "batch_size": 32,  # This is showing how many image will be load and process at one time
    "image_size": (256, 256),
    "seed": 1,
    "validation_split": .2,
    "class_names": breeds
}

train = tf.keras.utils.image_dataset_from_directory(
    "images",
    subset="training",
    **args
)

test = tf.keras.utils.image_dataset_from_directory(
    "images",
    subset="validation",
    **args
)

