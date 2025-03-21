import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import create_model, create_data_augmentation
import matplotlib.pyplot as plt

def train(fname, out_fname):
    try:
        with open(fname, 'r') as f:
            shipsnet = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл {fname} не найден.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Ошибка: Неверный формат JSON в файле {fname}.")
        sys.exit(1)

    # Preprocess image data and labels for input
    x = np.array(shipsnet['data']) / 255.0
    if x.size == 0 or len(x.shape) != 2:
        print("Ошибка: Некорректные данные в shipsnet['data'].")
        sys.exit(1)

    x = x.reshape([-1, 3, 80, 80]).transpose([0, 2, 3, 1])  # Convert to [None, 80, 80, 3]
    y = np.array(shipsnet['labels'])
    if y.size == 0 or len(y.shape) != 1:
        print("Ошибка: Некорректные метки в shipsnet['labels'].")
        sys.exit(1)
    y = tf.keras.utils.to_categorical(y, 2)  # One-hot encoding

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    # Create the model
    model = create_model()

    # Data augmentation and preprocessing
    datagen = create_data_augmentation()
    datagen.fit(x_train)  # Fit the data generator to compute statistics for normalization

    # Define Early Stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Мониторим валидационную потерю
        patience=10,         # Количество эпох без улучшений перед остановкой
        restore_best_weights=True  # Восстанавливаем веса лучшей модели
    )

    # Train the model
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=50,
        validation_data=(x_val, y_val),  # Use separate validation data
        callbacks=[early_stopping]       # Добавляем Early Stopping
    )

    # Save the trained model
    model.save(out_fname)
    print(f"Модель успешно сохранена в {out_fname}")

    # Plot training results
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plots to a file
    plt.tight_layout()
    plt.savefig('results/accuracy.png')  # Save the graphs to a file
    plt.close()


# Main function
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python train.py <input_file> <output_file>")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2])