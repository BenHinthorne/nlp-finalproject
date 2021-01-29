import numpy

## Computes Length of A Numpy Vector
def compute_length(a):
    return numpy.linalg.norm(a, axis=a.ndim-1)

## Computes cosine similarity of two numpy vectors
def cosine_similarity(array1, array2):
    dp = numpy.dot(array2, array1)
    cosine = dp/(compute_length(array1)*compute_length(array2))
    return cosine