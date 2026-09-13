import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X_np = np.array(X)
	y_np = np.array(y)

	theta = np.round(np.linalg.inv(np.matmul(X_np.T, X_np)) @ X_np.T @ y, 4)
	return theta