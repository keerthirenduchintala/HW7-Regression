"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor

def test_prediction():
	# Test the prediction 
    log_model = LogisticRegressor(num_feats=3, learning_rate=0.01, tol=0.01, max_iter=10, batch_size=10)
    log_model.W = np.array([1, -2, 3, -1])
    X = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 2]])
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    # Manually compute expected predictions using sigmoid on dot product
    # W·x: [1-2+3-1, 2+0+3-1, 0-6+6-1] = [1, 4, -1]
    z = np.array([1, 4, -1])
    y_true = 1 / (1 + np.exp(-z))
    y_pred = log_model.make_prediction(X)

    assert np.all(np.abs(y_pred - y_true) < 0.00001)


def test_loss_function():
	# Test the loss function
    log_model = LogisticRegressor(num_feats=3, learning_rate=0.01, tol=0.01, max_iter=10, batch_size=10)
    log_model.W = np.array([1, -2, 3, -1])
    X = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 2]])
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    # y_true derived from different raw values than the dot product
    y_true_raw = np.array([2, 5, 0])
    y_true = 1 / (1 + np.exp(-y_true_raw))
    y_pred = log_model.make_prediction(X)

    loss_result = log_model.loss_function(y_true, y_pred)
    true_loss = 0.430215876224504

    assert (np.abs(loss_result - true_loss) < 0.0001)


def test_gradient():
	# Test the gradient 
    log_model = LogisticRegressor(num_feats=3, learning_rate=0.012, tol=0.01, max_iter=10, batch_size=10)
    log_model.W = np.array([1, -2, 3, -1])
    X = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 2]])
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    y_true_raw = np.array([2, 5, 0])
    y_true = 1 / (1 + np.exp(-y_true_raw))

    gradient_result = log_model.calculate_gradient(y_true, X)
    gradient_true = np.array([-0.05744174, -0.28097141, -0.20771634, -0.13069681])

    assert (np.all(np.abs(gradient_result - gradient_true) < 0.0001))


def test_training():
	# Test training
    log_model = LogisticRegressor(num_feats=3, learning_rate=0.012, tol=0.01, max_iter=3, batch_size=2)

    X_train = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 2], [1, 2, 0]])
    y_train_raw = np.array([2, 5, 0, 3])
    y_train = 1 / (1 + np.exp(-y_train_raw))

    X_val = np.array([[1, 0, 1], [2, 1, 3]])
    y_val_raw = np.array([1, 4])
    y_val = 1 / (1 + np.exp(-y_val_raw))

    W_initial = np.copy(log_model.W)
    log_model.train_model(X_train, y_train, X_val, y_val)
    W_final = np.copy(log_model.W)

    assert np.all(W_initial != W_final)