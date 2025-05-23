import os
import shutil
import random
import tensorflow as tf

def setup_validation_split(training_dir, validation_dir, val_split=0.2, seed=42):
    random.seed(seed)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
        for class_name in os.listdir(training_dir):
            class_val_dir = os.path.join(validation_dir, class_name)
            if not os.path.exists(class_val_dir):
                os.makedirs(class_val_dir)
    for class_name in os.listdir(training_dir):
        class_train_dir = os.path.join(training_dir, class_name)
        class_val_dir = os.path.join(validation_dir, class_name)
        if os.path.isdir(class_train_dir):
            files = [f for f in os.listdir(class_train_dir) if os.path.isfile(os.path.join(class_train_dir, f))]
            random.shuffle(files)
            val_size = int(len(files) * val_split)
            val_files = files[:val_size]
            for file in val_files:
                src = os.path.join(class_train_dir, file)
                dst = os.path.join(class_val_dir, file)
                shutil.move(src, dst)

def create_tf_datasets(base_dir, image_size=(224, 224), batch_size=32):
    training_dir = os.path.join(base_dir, 'Training')
    validation_dir = os.path.join(base_dir, 'Validation')
    testing_dir = os.path.join(base_dir, 'Testing')
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        training_dir, label_mode="categorical", batch_size=batch_size, image_size=image_size)
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        validation_dir, label_mode="categorical", batch_size=batch_size, image_size=image_size)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        testing_dir, label_mode="categorical", image_size=image_size, shuffle=False)
    return train_data, val_data, test_data

def process_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image, label

def count_files_in_directory(directory):
    label_counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            label_counts[class_name] = len(files)
    return label_counts 