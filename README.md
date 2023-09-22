# Artificial Neural Network (ANN) Project

This README file provides an overview of the Artificial Neural Network (ANN) project. In this project, we'll create a neural network model to predict customer churn in a bank based on various customer attributes.

## Project Overview

The objective of this project is to build, train, and evaluate an ANN model to predict whether a customer will leave the bank (churn) or not. We will perform the following steps:

1. **Data Preprocessing**: We start by preprocessing the dataset, which includes loading the data, handling missing values, encoding categorical variables, and splitting the dataset into training and testing sets.

2. **Building the ANN**: We design the architecture of the neural network, specifying the number of layers and neurons in each layer, as well as the activation functions.

3. **Training the ANN**: We compile and train the neural network using the training data, specifying the loss function, optimizer, and metrics. We'll train the model for a certain number of epochs and with a specific batch size.

4. **Making Predictions**: After training, we use the model to make predictions on new data, including a specific example where we predict whether a customer will leave the bank based on provided attributes.

5. **Evaluating the Model**: We evaluate the model's performance using metrics such as accuracy and confusion matrix.

## Getting Started

Before running the code, make sure you have the required libraries installed, such as `numpy`, `pandas`, and `tensorflow`. You can install these dependencies using pip:

```bash
pip install numpy pandas tensorflow
```

## Usage

1. **Data Preprocessing**: The code begins with data preprocessing steps, including loading the dataset, encoding categorical variables, and splitting the data into training and testing sets.

2. **Building the ANN**: We create an ANN model using TensorFlow's Keras API. The model consists of input, hidden, and output layers with specified activation functions.

3. **Training the ANN**: The model is compiled with an optimizer, loss function, and evaluation metrics. It's then trained on the training data.

4. **Making Predictions**: We demonstrate how to make predictions using the trained model, including an example prediction for a specific customer.

5. **Evaluating the Model**: The code calculates the accuracy of the model and provides a confusion matrix to assess its performance on the test data.

## Example Prediction

To predict whether a customer will leave the bank based on the provided attributes, use the following code snippet:

```python
# Example Customer Data
customer_data = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]

# Make the prediction
prediction = ann.predict(sc.transform(customer_data))

# Check if the customer will leave the bank (output > 0.5)
will_leave = prediction > 0.5

print("Customer will leave the bank:", will_leave)
```

## Evaluation

The model's performance is evaluated using a confusion matrix and accuracy score. The confusion matrix provides insights into the model's true positive, true negative, false positive, and false negative predictions.

## Conclusion

This ANN project demonstrates how to build, train, and evaluate a neural network model for predicting customer churn in a bank. You can further fine-tune the model or apply it to your own datasets for similar classification tasks.
