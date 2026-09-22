import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	
	d = K.shape[1]
	
	s = (Q @ K.T) 
	s_scaled = s / np.sqrt(d)

	if mask is not None:
		s_scaled = np.where(mask == 0, -1e9, s_scaled)


	max_val = np.max(s_scaled, axis=1, keepdims=True)
	s_scaled = np.exp(s_scaled - max_val) / np.sum(np.exp(s_scaled - max_val), axis=1, keepdims=True)

	attention = s_scaled @ V
	

	
	return attention, s_scaled