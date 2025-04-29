from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

def create_img_df(train_path):
    # Get the path to project root (2 levels up from data.py)
    project_root = Path(__file__).resolve().parents[2]

    # Build paths to train directory
    dir = project_root / 'dataset' / 'raw' / train_path

    # Create dataframe from images
    brain_paths, brain_classes = [], []
    for brain_class in classes:
        class_brain = list((dir / brain_class).iterdir())

        brain_paths.extend(class_brain)
        brain_classes.extend([brain_class] * len(class_brain))

    df = pd.DataFrame({'Class Path': brain_paths, 'Class': brain_classes})

    return df

def create_tf_dataset(df):
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])

    file_pahts = df['Class Path'].values
    labels = df['Class'].values

    dataset = tf.data.Dataset.from_tensor_slices((file_pahts, labels))

    return dataset

# Function to load and preprocess each image
def process_image(path, label, img_size=(128, 128)):
    # Read image
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # or decode_png if needed
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    return image, label

if __name__ == "__main__":
    train_path = 'Training'

    # Create dataframe from images
    train_df = create_img_df(train_path)

    # Create TensorFlow dataset
    train_tf = create_tf_dataset(train_df)

    # Process images in the dataset
    train_tf = train_tf.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    