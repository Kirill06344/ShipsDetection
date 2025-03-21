"""
Apply trained machine learning model to an entire image scene using a sliding window.
"""

import sys
import os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, center_of_mass, label
import tensorflow as tf


def detector(model_fname, in_fname, out_fname=None):
    """
    Perform a sliding window detector on an image.

    Args:
        model_fname (str): Path to Tensorflow model file (.h5 or .tflite)
        in_fname (str): Path to input image file
        out_fname (str): Path to output image file. Default is None.
    """

    # Load the trained model
    try:
        model = tf.keras.models.load_model(model_fname)
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)

    # Read input image data
    try:
        im = Image.open(in_fname).convert("RGB")  # Ensure RGB format
        arr = np.array(im) / 255.0  # Normalize pixel values to [0, 1]
        shape = arr.shape
        print(f"Изображение успешно загружено. Размер: {shape}")
    except FileNotFoundError:
        print(f"Ошибка: Файл {in_fname} не найден.")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении изображения: {e}")
        sys.exit(1)

    # Set output filename
    if not out_fname:
        out_fname = os.path.splitext(in_fname)[0] + '_detection.png'

    # Create detection variables
    detections = np.zeros((shape[0], shape[1]), dtype='uint8')
    output = np.copy(arr * 255).astype(np.uint8)  # Convert back to uint8 for drawing

    # Sliding window parameters
    step = 10  # Увеличенный шаг
    win = 80
    batch_size = 64  # Размер батча

    # Собираем все фрагменты в один список
    chips = []
    positions = []  # Сохраняем позиции для каждого фрагмента

    print("Обработка изображения...")
    for i in range(0, shape[0] - win, step):
        print(f"Обработка строки {i} из {shape[0] - win}")

        for j in range(0, shape[1] - win, step):
            chip = arr[i:i + win, j:j + win, :]
            chips.append(chip)
            positions.append((i, j))

            # Если набрался батч, делаем предсказание
            if len(chips) == batch_size:
                predictions = model.predict(np.array(chips), verbose=0)
                predicted_labels = np.argmax(predictions, axis=1)

                # Обновляем detections
                for (i, j), label in zip(positions, predicted_labels):
                    if label == 1:
                        detections[i + win // 2, j + win // 2] = 1

                # Очищаем батч
                chips = []
                positions = []

    # Обрабатываем оставшиеся фрагменты
    if chips:
        predictions = model.predict(np.array(chips), verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        for (i, j), label in zip(positions, predicted_labels):
            if label == 1:
                detections[i + win // 2, j + win // 2] = 1

    # Process detection locations
    dilation = binary_dilation(detections, structure=np.ones((3, 3)))
    labels, n_labels = label(dilation)
    centers_of_mass = center_of_mass(dilation, labels, np.arange(n_labels) + 1)

    # Draw bounding boxes around detected regions
    if isinstance(centers_of_mass, tuple):  # Handle single detection
        centers_of_mass = [centers_of_mass]

    for i, j in centers_of_mass:
        i = int(i - win / 2)
        j = int(j - win / 2)

        # Ensure bounding box stays within image bounds
        i = max(0, i)
        j = max(0, j)
        i_end = min(shape[0], i + win)
        j_end = min(shape[1], j + win)

        # Draw bounding box
        output[i:i_end, j:j + 2, :] = [255, 0, 0]  # Left edge
        output[i:i_end, j_end - 2:j_end, :] = [255, 0, 0]  # Right edge
        output[i:i + 2, j:j_end, :] = [255, 0, 0]  # Top edge
        output[i_end - 2:i_end, j:j_end, :] = [255, 0, 0]  # Bottom edge

    # Save output image
    try:
        out_im = Image.fromarray(output)
        out_im.save(out_fname)
        print(f"Результат сохранен в файл: {out_fname}")
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")


# Main function
if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Использование: python detector.py <model_file> <input_image> [output_image]")
        sys.exit(1)

    model_fname = sys.argv[1]
    in_fname = sys.argv[2]
    out_fname = sys.argv[3] if len(sys.argv) == 4 else None

    detector(model_fname, in_fname, out_fname)