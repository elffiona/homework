import numpy as np


class LinearRegression:
    """
    A linear regression model that uses close form solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X: np.ndarray, Features
        y: np.ndarray, Label
        Use close form solution to calculate weights and bias term
        """
        n, p = X.shape
        bias_col = np.ones([n, 1])
        train_data_new = np.hstack((X, bias_col))
        # X^TX
        tmp = np.matmul(train_data_new.T, train_data_new)
        # (X^TX)^-1
        inv = np.linalg.inv(tmp)
        # (X^TX)^-1 X^T y
        tmp = np.matmul(np.matmul(inv, train_data_new.T), np.array([y]).T)

        self.w = tmp[:-1]
        self.b = tmp[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X : np.ndarray
        Use matrix multiplication to calculate the prediction
        """
        pred_train = np.matmul(X, self.w).reshape((X.shape[0], 1))
        bias_train = self.b * np.ones((X.shape[0], 1))
        pred_final = pred_train + bias_train
        return pred_final


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    w: np.ndarray
    b: float

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        X: np.ndarray, Features
        y: np.ndarray, Label
        lr: float, Learning rate
        epochs: int, Number of training epochs
        Use gradient descent to update the weights and bias term
        """
        n, p = X.shape
        # Initialize the weight and bias term with 0
        self.w = np.zeros((p, 1))
        self.b = 0.0
        # Start iterating
        for i in range(epochs):
            # Calculate gradient
            pred = self.predict(X)
            dw = (np.matmul(X.T, (np.array([y]).T - pred)) * 2) / n
            db = (np.sum(np.array([y]).T - pred) * 2) / n
            # Update
            self.w = self.w + dw * lr
            self.b = self.b + db * lr

            # # trial for regular
            # bias_col_1 = np.ones([n, 1])
            # train_data_new_1 = np.hstack((X, bias_col_1))
            # # X^TX
            # tmp_1 = np.matmul(train_data_new_1.T, train_data_new_1)
            # # (X^TX)^-1
            # inv_1 = np.linalg.inv(tmp_1)
            # # (X^TX)^-1 X^T y
            # tmp_1 = np.matmul(np.matmul(inv_1, train_data_new_1.T), np.array([y]).T)
            #
            # print("Shape of closed form w:")
            # print(tmp_1[:-1].shape)
            # print("Shape of closed form b:")
            # print(tmp_1[-1].shape)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        n, p = X.shape
        pred_train = np.matmul(X, self.w).reshape((n, 1))
        bias_train = self.b * np.ones((n, 1))
        pred_final = pred_train + bias_train
        return pred_final
