import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Reshape, BatchNormalization, Flatten, Dense
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from sklearn.decomposition import PCA
import tensorflow as tf

def Combined_model(image_size=(224, 224, 3), n_classes=4, init_lr=1e-3):
    input_layer = Input(shape=image_size)
    baseModel = VGG19(weights='imagenet', include_top=False, input_tensor=input_layer)
    for layer in baseModel.layers:
        layer.trainable = False
    x = baseModel.output
    x = Reshape((49, 512))(x)
    x = LSTM(512, activation="relu", return_sequences=True, trainable=False)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    opt = Adam(learning_rate=init_lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model

def extract_features(data_generator, model):
    features = []
    labels = []
    for images, batch_labels in data_generator:
        batch_features = model.predict(images)
        features.append(batch_features)
        labels.append(batch_labels.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def apply_pca(features, n_components=64):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    return features_pca, pca 