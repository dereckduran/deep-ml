import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    # Building design matrix
    # Need to take n numbers from X train
    # Multiply it from 0 to degree
    # put it into a matrix

    train_errors = []
    val_errors = []
    for n in train_sizes:

        matrix_rows = []
        
        samples = X_train[:n]
        
        for x in samples:
            powers = []
            for deg in range(0, degree + 1):
                powers.append(x[0] ** deg)
            matrix_rows.append(powers)

        power_matrix = np.array(matrix_rows)

        weights = np.linalg.pinv(power_matrix) @ y_train[:n]

        val_matrix = []
        for x in X_val:
            powers = []
            for deg in range(0, degree + 1):
                powers.append(x[0] ** deg)
            val_matrix.append(powers)

        train_preds = power_matrix @ weights
        val_preds = np.array(val_matrix) @ weights

        train_mse = np.square(np.subtract(y_train[:n], train_preds)).mean()
        val_mse = np.square(np.subtract(y_val, val_preds)).mean()

        train_errors.append(train_mse)
        val_errors.append(val_mse)
    
    if train_errors[-1] > bias_threshold:
        return {'train_errors': train_errors, 'val_errors': val_errors, 'diagnosis': 'high_bias'}

    elif (val_errors[-1] - train_errors[-1]) > variance_threshold:
        return {'train_errors': train_errors, 'val_errors': val_errors, 'diagnosis': 'high_variance'}
    else:
        return {'train_errors': train_errors, 'val_errors': val_errors, 'diagnosis': 'good_fit'}
