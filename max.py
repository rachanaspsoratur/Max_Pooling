import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

# Max Pooling Algorithm
def max_pooling(matrix, k):
    m, n = matrix.shape
    output = np.zeros((m - k + 1, n - k + 1))
    for i in range(m - k + 1):
        for j in range(n - k + 1):
            window = matrix[i:i+k, j:j+k]
            output[i, j] = np.max(window)
    return output

# Improved Max Pooling Algorithm
def improved_max_pooling(matrix, k):
    m, n = matrix.shape
    output = np.zeros((m - k + 1, n - k + 1))
    for i in range(m - k + 1):
        for j in range(n - k + 1):
            window = matrix[i:i+k, j:j+k]
            max_val = -float('inf')
            for x in range(k):
                max_val = max(max_val, max(window[x]))
            output[i, j] = max_val
    return output

    # Max Pooling Algorithm
    matrix = np.random.randint(0, 10, size=(5, 5))
    k = 3
    output = max_pooling(matrix, k)
    print(output)

    # Improved Max Pooling Algorithm
    improved_output = improved_max_pooling(matrix, k)
    print(improved_output)