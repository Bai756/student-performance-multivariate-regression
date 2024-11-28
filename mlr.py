import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Student_Performance.csv')
dataset['Extracurricular Activities'] = dataset['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

dataset = (dataset - dataset.mean()) / dataset.std()

X = dataset.iloc[:, :-1].values
ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)

y = dataset.iloc[:, -1].values
theta = np.zeros([6, 1])

alpha = 0.01
iterations = 100

def computeCost(X, y, theta):
    to_be_summed = np.power(((X @ theta) - y), 2)
    return np.sum(to_be_summed) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iterations):
    cost = np.zeros(iterations)
    alpha_div_len = alpha / len(X)
    
    for i in range(iterations):
        error = X @ theta - y
        theta = theta - alpha_div_len * (X.T @ error)
        cost[i] = computeCost(X, y, theta)
        if i % 10 == 0:
            print('Iteration:', i, 'Cost:', cost[i])
    return (theta, cost)

print('Initial Cost:', computeCost(X, y, theta))

g, cost = gradientDescent(X, y, theta, alpha, iterations)

# plot the cost function
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], y, color='blue', label='Hours Studied')
plt.scatter(X[:, 2], y, color='green', label='Previous Scores')
plt.scatter(X[:, 3], y, color='red', label='Extracurricular Activities')
plt.scatter(X[:, 4], y, color='purple', label='Sleep Hours')
plt.scatter(X[:, 5], y, color='orange', label='Sample Question Papers Practiced')
plt.xlabel('Feature Value')
plt.ylabel('Score')
plt.title('Student Performance')
