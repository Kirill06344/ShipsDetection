import sys
import json
import numpy as np
import tensorflow as tf
from model import create_model, create_data_augmentation

def train(fname, out_fname):
    """ Train and save CNN model on ShipsNet dataset """
    # Load shipsnet data
    with open(fname, 'r') as f:
        shipsnet = json.load(f)

    # Preprocess image data and labels for input
    x = np.array(shipsnet['data']) / 255.0
    x = x.reshape([-1, 3, 80, 80]).transpose([0, 2, 3, 1])  # Convert to [None, 80, 80, 3]
    y = np.array(shipsnet['labels'])
    y = tf.keras.utils.to_categorical(y, 2)  # One-hot encoding

    # Create the model
    model = create_model()

    # Data augmentation and preprocessing
    datagen = create_data_augmentation()
    datagen.fit(x)  # Fit the data generator to compute statistics for normalization

    # Train the model
    model.fit(datagen.flow(x, y, batch_size=128),
              epochs=50,
              validation_split=0.2)

    # Save the trained model
    model.save(out_fname)


# Main function
if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])