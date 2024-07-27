import os
import numpy as np
import cv2
import tensorflow as tf


def load_data(data_dir):
    """
    Load images and labels from directory.

    Args:
    data_dir (str): Path to the data directory.

    Returns:
    images (list): List of images.
    labels (list): List of corresponding labels.
    """
    images = []
    labels = []
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for idx, emotion in enumerate(classes):
        emotion_dir = os.path.join(data_dir, emotion)
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(idx)
    return np.array(images), np.array(labels)


def preprocess_data(images, labels):
    """
    Preprocess images and labels.

    Args:
    images (numpy array): Array of images.
    labels (numpy array): Array of labels.

    Returns:
    preprocessed_images (numpy array): Preprocessed images.
    one_hot_labels (numpy array): One-hot encoded labels.
    """
    images = images / 255.0
    images = np.expand_dims(images, -1)
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=7)
    return images, one_hot_labels


if __name__ == "__main__":
    train_images, train_labels = load_data('../data/train')
    test_images, test_labels = load_data('../data/test')

    train_images, train_labels = preprocess_data(train_images, train_labels)

    test_images, test_labels = preprocess_data(test_images, test_labels)

    np.save('../data/train_images.npy', train_images)
    np.save('../data/train_labels.npy', train_labels)
    np.save('../data/test_images.npy', test_images)
    np.save('../data/test_labels.npy', test_labels)
