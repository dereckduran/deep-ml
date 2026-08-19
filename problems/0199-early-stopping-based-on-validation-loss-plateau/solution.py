import numpy as np
def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
	"""
	Determine at each epoch whether training should stop based on validation loss.
	
	Args:
		val_losses: List of validation losses at each epoch
		patience: Number of epochs to wait for improvement before stopping
		min_delta: Minimum change in validation loss to qualify as improvement
	
	Returns:
		List of booleans indicating whether to stop at each epoch
	"""
	# Your code here
	best_loss = np.inf
	earlies = []
	counter = 0
	for i in range(len(val_losses)):
		loss = val_losses[i]

		# if delta isnt over min improvement then increment counter
		if loss < best_loss - min_delta:
			best_loss = loss
			counter = 0
			earlies.append(False)

		else:
			counter += 1
			earlies.append(counter >= patience)
	return earlies
		
