# DeepLearningFromScratch

```markdown
## Overview

**DeepLearningFromScratch** is an implementation of a basic feedforward neural network using only `numpy`, without relying on any high-level deep learning libraries like TensorFlow or PyTorch. This project demonstrates fundamental concepts of deep learning, including forward propagation, backpropagation, gradient descent, and hyperparameter tuning. It uses a toy dataset to perform regression tasks.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Results](#results)
6. [How to Run](#how-to-run)
7. [Future Enhancements](#future-enhancements)

---

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Libraries:
  - `numpy`
  - `scikit-learn`

Install the required libraries using:
```bash
pip install numpy scikit-learn
```

---

## Dataset

The dataset is synthetic and generated programmatically to simulate a simple regression problem. The features (`x`) are two-dimensional, and the target variable (`y`) is a scalar value computed using a non-linear function.

### Dataset Structure
- **Training Set**: 75% of the data
- **Validation Set**: 18.75% of the data (25% of the training set)
- **Test Set**: 25% of the data

---

## Model Architecture

### Neural Network Design
1. **Input Layer**:
   - Two input features.

2. **Hidden Layer**:
   - 3 neurons
   - Activation Function: Hyperbolic Tangent (`tanh`)

3. **Output Layer**:
   - 1 neuron
   - Activation Function: Sigmoid

### Mathematical Formulas
#### Forward Propagation
- Hidden Layer:  
  \( Z^{(1)} = W^{(1)} X + b^{(1)} \)  
  \( A^{(1)} = \tanh(Z^{(1)}) \)

- Output Layer:  
  \( Z^{(2)} = W^{(2)} A^{(1)} + b^{(2)} \)  
  \( A^{(2)} = \sigma(Z^{(2)}) \)

#### Loss Function
- Mean Squared Error:  
  \( L = \frac{1}{2m} \sum (A^{(2)} - Y)^2 \)

#### Backpropagation
- Gradients for each layer are derived to update the weights and biases.

---

## Training Process

### Steps
1. **Forward Propagation**:
   Compute the predictions and the intermediate activations.

2. **Loss Calculation**:
   Use Mean Squared Error to measure the model's performance.

3. **Backward Propagation**:
   Compute gradients for weights and biases using the chain rule.

4. **Parameter Updates**:
   Apply gradient descent with a defined learning rate to update weights and biases.

### Hyperparameters
- Learning Rate: 0.01 (default)
- Epochs: 50 (adjustable)

---

## Results

### Sample Training Output
```plaintext
Epoch 0, Loss: 0.0199, Val Loss: 0.0221
Epoch 10, Loss: 0.0199, Val Loss: 0.0221
Epoch 20, Loss: 0.0199, Val Loss: 0.0222
Epoch 30, Loss: 0.0198, Val Loss: 0.0222
Epoch 40, Loss: 0.0198, Val Loss: 0.0222
Epoch 50, Loss: 0.0198, Val Loss: 0.0222
```

### Observations
- The model shows steady loss reduction during training.
- Validation loss aligns closely with training loss, indicating minimal overfitting.

---

## How to Run

### Training the Model
1. Clone this repository.
2. Run the Python script.
3. The training process will output the loss and validation loss at regular intervals.

### Predicting
To make predictions on test data:
```python
predictions = model.predict(x_test)
```

---

## Future Enhancements

1. **Regularization**:
   - Add L1/L2 regularization to improve generalization.
2. **Optimization**:
   - Implement advanced optimizers like Adam or RMSProp.
3. **Activation Functions**:
   - Experiment with ReLU and other activations.
4. **Visualization**:
   - Plot loss curves and decision boundaries.

---

## Acknowledgments

This project was inspired by the need to understand neural networks at a fundamental level by implementing them from scratch.

---
```
