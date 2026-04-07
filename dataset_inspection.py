import numpy as np
import graphlearning as gl

X = np.load("data/mnistimb_vae.npz")['data']
y = np.load("data/mnistimb_labels.npz")['labels']
X2= np.load("data/mnist_vae.npz")['data']
y2= np.load("data/mnist_labels.npz")['labels']
X3= np.load("data/mnist_raw.npz")['data']

print(X.shape)
print(y.shape)
print(X2.shape)
print(y2.shape)
print(X3.shape)
def print_class_counts(name, y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{name} class counts:")
    for u, c in zip(unique, counts):
        print(f"  class {u}: {c}")

y_mod3 = y % 3
y2_mod3 = y2 % 3
print_class_counts("mnistimb", y_mod3)
print_class_counts("mnist", y2_mod3)

print()
X, clusters = gl.datasets.load('mnist', metric='vae')
print(X.shape)
print(clusters.shape)
print(set(clusters))