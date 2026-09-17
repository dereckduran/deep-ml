def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	import numpy as np
	def sigmoid(x):
		sig = 1 / (1 + np.exp(x))
		return sig * (1 - sig)

	def tanh(x):
		th = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) 
		return 1 - th**2

	def relu(x):

		return 1 if x > 0 else 0

	return {"sigmoid": sigmoid(x), "tanh": tanh(x), "relu": relu(x)}