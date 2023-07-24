from numpy.linalg import norm
import numpy as np

def cosine(x, y):
    return np.dot(x,y)/(norm(x)*norm(y))