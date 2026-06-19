import math
import random
import numpy as np

# Dot Product

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


# Magnitude (Length of vector)

def magnitude(v):
    return math.sqrt(sum(x * x for x in v))


# Cosine Similarity
def cosine_similarity(a, b):
    return dot_product(a, b) / (magnitude(a) * magnitude(b))


# Mean

def mean(v):
    return sum(v) / len(v)


# Variance

def variance(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / len(v)


# Testing against NumPy
def test():
    for _ in range(10):
        a = [random.uniform(-10, 10) for _ in range(5)]
        b = [random.uniform(-10, 10) for _ in range(5)]

        # Dot product test
        assert abs(dot_product(a, b) - np.dot(a, b)) < 1e-9

        # Magnitude test
        assert abs(magnitude(a) - np.linalg.norm(a)) < 1e-9

        # Cosine similarity test
        assert abs(cosine_similarity(a, b) -
                   (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))) < 1e-9

        # Mean test
        assert abs(mean(a) - np.mean(a)) < 1e-9

        # Variance test
        assert abs(variance(a) - np.var(a)) < 1e-9

    print("All tests passed!")


if __name__ == "__main__":
    test()