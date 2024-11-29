import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

alpha = 0.01
iterations = 1000

def compute_cost(X, y, theta):
    to_be_summed = np.power(((X @ theta) - y), 2)
    return np.sum(to_be_summed) / (2 * len(X))

def gradient_descent(X, y, theta, alpha, iterations):
    cost = np.zeros(iterations)
    alpha_over_m = alpha / len(X)
    theta_history = np.zeros((iterations, theta.shape[0]))
    
    for i in range(iterations):
        error = X @ theta - y
        theta = theta - alpha_over_m * (X.T @ error)
        cost[i] = compute_cost(X, y, theta)
        theta_history[i, :] = theta.flatten()
        if i % 100 == 0:
            print(f'Iteration: {i}, Cost: {cost[i]}')
    return theta, cost, theta_history


ori_dataset = pd.read_csv('Student_Performance.csv')
dataset = ori_dataset.copy()
dataset['Extracurricular Activities'] = ori_dataset['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

normalized_data = (dataset - dataset.mean()) / dataset.std()

X = normalized_data.iloc[:, :-1].values
ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)

y = normalized_data.iloc[:, -1].values.reshape(-1, 1)
theta = np.zeros([6, 1])

theta, cost, theta_history = gradient_descent(X, y, theta, alpha, iterations)


# plot the cost
plt.plot(range(iterations), cost, 'r')
plt.title('Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# plot change in theta values
feature_names = ['Intercept'] + list(ori_dataset.columns[:-1])  # Add intercept to the list of features
plt.figure(figsize=(10, 6))
for i in range(theta_history.shape[1]):
    plt.plot(range(iterations), theta_history[:, i], label=f'Theta {i} ({feature_names[i]})')
plt.xlabel('Iterations')
plt.ylabel('Theta Value')
plt.title('Change in Theta During Gradient Descent')
plt.legend(loc='best')
plt.show()

# Plot predictions
predictions = X @ theta
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.6, color='blue')
plt.plot(y, y, color='red', linestyle='--', linewidth=2, label='Perfect Prediction') 
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()


# Plot the data in comparison to extracurricular activities
fig, axs = plt.subplots(2, 2, figsize=(15, 7)) 

sns.scatterplot(
    ax=axs[0, 0],
    data=ori_dataset,
    x='Hours Studied',
    y='Performance Index',
    hue='Extracurricular Activities',
    palette='Set1',
    alpha=0.7
)
sns.regplot(
    ax=axs[0, 0],
    data=ori_dataset,
    x='Hours Studied',
    y='Performance Index',
    scatter=False,
    color='blue',
    line_kws={"linewidth": 2},
)
axs[0, 0].set_title('Hours Studied vs Scores')

sns.scatterplot(
    ax=axs[0, 1],
    data=ori_dataset,
    x='Sleep Hours',
    y='Performance Index',
    hue='Extracurricular Activities',
    palette='Set2',
    alpha=0.7
)
sns.regplot(
    ax=axs[0, 1],
    data=ori_dataset,
    x='Sleep Hours',
    y='Performance Index',
    scatter=False,
    color='purple',
    line_kws={"linewidth": 2},
)
axs[0, 1].set_title('Sleep Hours vs Scores')

sns.scatterplot(
    ax=axs[1, 0],
    data=ori_dataset,
    x='Previous Scores',
    y='Performance Index',
    hue='Extracurricular Activities',
    palette='Set3',
    alpha=0.7
)
sns.regplot(
    ax=axs[1, 0],
    data=ori_dataset,
    x='Previous Scores',
    y='Performance Index',
    scatter=False,
    color='green',
    line_kws={"linewidth": 2},
)
axs[1, 0].set_title('Previous Scores vs Scores')

sns.scatterplot(
    ax=axs[1, 1],
    data=ori_dataset,
    x='Sample Question Papers Practiced',
    y='Performance Index',
    hue='Extracurricular Activities',
    palette='Set2',
    alpha=0.7
)
sns.regplot(
    ax=axs[1, 1],
    data=ori_dataset,
    x='Sample Question Papers Practiced',
    y='Performance Index',
    scatter=False,
    color='orange',
    line_kws={"linewidth": 2},
)
axs[1, 1].set_title('Sample Question Papers Practiced vs Scores')

plt.tight_layout()
plt.show()