import os
import datetime
from time import time

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input

from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, AvgPool2D, Dropout, BatchNormalization, GlobalAveragePooling2D, Lambda, multiply
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np
import data_preparation
import params
from utils import plot_train_metrics, save_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')

# Load the metadata and preprocess it
metadata = data_preparation.load_metadata()
metadata, labels = data_preparation.preprocess_metadata(metadata)
train, valid, test = data_preparation.stratify_train_test_split(metadata)

def create_data_generator(dataset, labels, batch_size, preprocessing_function, color_mode="rgb", target_size=params.IMG_SIZE):
    """
    Creates a Keras DataGenerator for the input dataset.

    Args:
      dataset: The images subset to use.
      labels: The labels to use.
      batch_size: The batch size of the generator.
      color_mode: one of "grayscale", "rgb". Default: "rgb".
      target_size: The (x, y) image size to scale the images.

    Returns:
      The created ImageDataGenerator.
    """
    dataset['newLabel'] = dataset.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

    image_generator = ImageDataGenerator(samplewise_center=True,
                                         samplewise_std_normalization=True,
                                         horizontal_flip=True,
                                         vertical_flip=False,
                                         height_shift_range=0.05,
                                         width_shift_range=0.1,
                                         rotation_range=5,
                                         shear_range=0.1,
                                         fill_mode='reflect',
                                         zoom_range=0.15,
                                         preprocessing_function=preprocessing_function)

    dataset_generator = image_generator.flow_from_dataframe(dataframe=dataset,
                                                            directory=None,
                                                            x_col='path',
                                                            y_col='newLabel',
                                                            class_mode='categorical',
                                                            classes=labels,
                                                            target_size=target_size,
                                                            color_mode=color_mode,
                                                            batch_size=batch_size)

    return dataset_generator

def _create_attention_model(frozen_model, labels, optimizer='adam'):
    """
    Creates an attention model to train on a pre-trained model's output features.

    Args:
      frozen_model: The pre-trained model (e.g., VGG19).
      labels: The labels to use.
      optimizer: The optimizer to use.

    Returns:
      The created Model.
    """
    frozen_features = Input(frozen_model.output_shape[1:], name='feature_input')
    frozen_depth = frozen_model.output_shape[-1]
    new_features = BatchNormalization()(frozen_features)

    # Attention mechanism
    attention_layer = Conv2D(128, kernel_size=(1, 1), padding='same', activation='elu')(new_features)
    attention_layer = Conv2D(32, kernel_size=(1, 1), padding='same', activation='elu')(attention_layer)
    attention_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='elu')(attention_layer)
    attention_layer = AvgPool2D((2, 2), strides=(1, 1), padding='same')(attention_layer)  # Smooth results
    attention_layer = Conv2D(1, kernel_size=(1, 1), padding='valid', activation='sigmoid')(attention_layer)

    # Fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, frozen_depth))
    up_c2 = Conv2D(frozen_depth, kernel_size=(1, 1), padding='same', activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attention_layer = up_c2(attention_layer)

    mask_features = multiply([attention_layer, new_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attention_layer)

    # To account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.5)(gap)
    dr_steps = Dropout(0.5)(Dense(128, activation='elu')(gap_dr))
    out_layer = Dense(len(labels), activation='sigmoid')(dr_steps)

    # Creating the final model
    attention_model = Model(inputs=[frozen_features], outputs=[out_layer], name='attention_model')
    attention_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return attention_model

def _create_base_model(Model, labels, input_shape, trainable=False, weights="imagenet"):
    """
    Creates a Keras base model for transfer learning.

    Args:
      Model: The Keras class to initialize.
      labels: The labels to use.
      input_shape: The shape of the Network input.
      trainable: Is the model able to be trained?
      weights: Which pre-trained weights to use if any.

    Returns:
      The created Model.
    """
    base_model = Model(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = trainable
    return base_model

def create_attention_model(base_model, labels, optimizer='adam'):
    """
    Creates an attention model by adding attention layers to the base model.

    Args:
      base_model: The Keras Base Model to start with.
      labels: The labels to use.
      optimizer: The optimizer to use.

    Returns:
      The created attention Model.
    """
    attention_model = _create_attention_model(base_model, labels, optimizer=optimizer)

    model = Sequential(name='combined_model')
    model.add(base_model)
    model.add(attention_model)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    print(f'{model.summary()}')
    return model

def fit_model(model, model_name, train, valid):
    """
    Fits the model.

    Args:
      model: The model to train.
      train: The training data generator.
      valid: The validation data generator.
    """
    results_folder = os.path.join(params.RESULTS_FOLDER, f'run-{RUN_TIMESTAMP}')
    os.makedirs(results_folder, exist_ok=True)

    # Callbacks for TensorBoard logging, model checkpointing, and early stopping
    tensorboard = TensorBoard(log_dir=results_folder)
    model_checkpoint = ModelCheckpoint(
        os.path.join(results_folder, f'best_model_{model_name}.h5'),
        monitor='val_loss', save_best_only=True
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(
        train,
        validation_data=valid,
        steps_per_epoch=len(train),
        validation_steps=len(valid),
        epochs=params.EPOCHS,
        callbacks=[tensorboard, model_checkpoint, early_stopping, reduce_lr],
    )

    save_model(model, results_folder, model_name)
    plot_train_metrics(history, results_folder, model_name)

# Example Usage
# Create data generators for training and validation
train_data = create_data_generator(train, labels, params.BATCH_SIZE, VGG19_preprocess_input)
valid_data = create_data_generator(valid, labels, params.BATCH_SIZE, VGG19_preprocess_input)

# Create and train the model
base_model = _create_base_model(VGG19, labels, (params.VGG19_IMG_SIZE[0], params.VGG19_IMG_SIZE[1], 3), trainable=False)
attention_model = create_attention_model(base_model, labels)

fit_model(attention_model, "VGG19_attention", train_data, valid_data)
