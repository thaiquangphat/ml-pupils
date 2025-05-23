from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from math import floor

EPOCHS = 1
INIT_LR = 1e-3
decay_rate = 0.95
decay_step = 1
IMAGE_SIZE = [224, 224]


def get_callbacks(save_model_path):
    checkpoint = ModelCheckpoint(filepath=save_model_path,
                                 monitor='val_accuracy',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)
    earlystop = EarlyStopping(monitor='accuracy',
                              min_delta=0,
                              patience=15,
                              verbose=1,
                              mode='max')
    lr_scheduler = LearningRateScheduler(lambda epoch: INIT_LR * pow(decay_rate, floor(epoch / decay_step)))
    return [checkpoint, earlystop, lr_scheduler] 