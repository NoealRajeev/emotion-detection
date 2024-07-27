# Emotion Detection from Facial Expressions

This project detects human emotions from facial expressions using a Convolutional Neural Network (CNN) model. The project uses the FER-2013 dataset for training and testing and includes a real-time emotion detection system using a webcam.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Real-Time Emotion Detection](#real-time-emotion-detection)
- [Directory Structure](#directory-structure)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/NoealRajeev/emotion-detection.git
    cd emotion-detection
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

The project uses the FER-2013 dataset. You can download it from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). After downloading, place the dataset in the `data/` directory following the structure described in the [Directory Structure](#directory-structure) section.

## Data Preparation

The data preparation script preprocesses the dataset by resizing the images and normalizing the pixel values. Run the script to prepare the data:

```sh
  python src/data_preparation.py
```

This will generate train_images.npy, train_labels.npy, test_images.npy, and test_labels.npy files in the data/ directory.

 ## Training the Model
To train the CNN model, run the training script:
```sh
  python src/train_model.py
```
The trained model will be saved in the models/ directory as emotion_detection_model.h5.

## Real-Time Emotion Detection
To run the real-time emotion detection using your webcam, execute the following script:
```sh
  python src/real_time_detection.py
```
This will open a window showing the webcam feed with the predicted emotion displayed.

## Directory Structure
```md
emotion-detection/
├── data/
│   ├── test/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   ├── surprise/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   ├── surprise/
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── test_images.npy
│   ├── test_labels.npy
├── models/
│   └── emotion_detection_model.h5
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── train_model.py
│   └── real_time_detection.py
├── .gitignore
├── README.md
└── requirements.txt
```
Feel free to fork this project, raise issues, or contribute to it. Happy coding!

## Acknowledgements
- [Kaggle](https://www.kaggle.com/) for providing the FER-2013 dataset.
- [TensorFlow](https://www.tensorflow.org/) for the machine learning framework.

### Explanation

- **Installation**: Instructions to clone the repository, set up a virtual environment, and install dependencies.
- **Dataset**: Information on where to download the dataset and how to place it in the directory.
- **Data Preparation**: Instructions to run the data preparation script.
- **Training the Model**: Steps to train the CNN model.
- **Real-Time Emotion Detection**: Steps to run the real-time emotion detection script.
- **Directory Structure**: Detailed layout of the project directory.
- **Acknowledgements**: Credits to sources and tools used in the project.
