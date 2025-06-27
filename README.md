
# Digit Recognition with KNN

This project demonstrates a simple implementation of handwritten digit recognition using the K-Nearest Neighbors (KNN) algorithm. The dataset used for this project is the MNIST dataset, which consists of images of handwritten digits from 0 to 9.

## Project Description

The goal of this project is to classify the images of handwritten digits using the KNN algorithm. The KNN algorithm is a simple, yet powerful machine learning algorithm that works by classifying a data point based on the majority class of its neighbors.

### Steps

1. **Data Loading**: The MNIST dataset is loaded and split into training and testing data.
2. **Data Preprocessing**: The images are preprocessed to fit the input format of the KNN model.
3. **Model Training**: A KNN classifier is trained using the training data.
4. **Prediction**: The trained model is used to predict the class of a given test image.
5. **Evaluation**: The performance of the model is evaluated using metrics such as accuracy.

### Technologies Used

- Python
- TensorFlow (Keras)
- Scikit-learn
- NumPy
- Matplotlib

### Requirements

To run this project, you'll need to install the required Python libraries. You can do this using `pip`:

```bash
pip install tensorflow scikit-learn numpy matplotlib
