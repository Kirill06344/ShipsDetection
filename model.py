"""
Define Convolutional Neural Network model for ShipsNet input using TensorFlow
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Real-time data preprocessing and augmentation
def create_data_augmentation():
    datagen = ImageDataGenerator(
        featurewise_center=True,          # Zero-center the data
        featurewise_std_normalization=True,  # Normalize the data
        horizontal_flip=True,             # Randomly flip images horizontally
        vertical_flip=True,               # Randomly flip images vertically
        rotation_range=25                 # Randomly rotate images by up to 25 degrees
    )
    return datagen


# Define the CNN model
def create_model():
    model = Sequential([
        # Input layer with shape [80, 80, 3]
        Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
