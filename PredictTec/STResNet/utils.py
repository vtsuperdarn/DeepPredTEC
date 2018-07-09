import numpy as np

def batch_generator(X, y, batch_size):
    """Primitive batch generator 
    """
    size = X.shape[0]
    print (size)
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            #print "inside hi"
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue

def unshuffle_batch_generator(X, y, batch_size):
    """Primitive batch generator 
    """
    size = X.shape[0]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            continue

