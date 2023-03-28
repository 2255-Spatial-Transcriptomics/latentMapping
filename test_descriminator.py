import matplotlib.pyplot as plt 
from models.descriminatorModels.classifier_trainer import BinaryClassifierTrainer
import numpy as np
discriminator = BinaryClassifierTrainer(n_latent=2)

num_samples = 1000
mean_0 = np.array([0, 0])
cov_0 = np.array([[1, 0.5], [0.5, 1]])  # covariance matrix for class 0
mean_1 = np.array([2, 2])
cov_1 = np.array([[1, -0.5], [-0.5, 1]])  # covariance matrix for class 1
X_0 = np.random.multivariate_normal(mean_0, cov_0, num_samples // 2)
X_1 = np.random.multivariate_normal(mean_1, cov_1, num_samples // 2)
X = np.concatenate((X_0, X_1))
y = np.concatenate((np.zeros(num_samples // 2), np.ones(num_samples // 2)))

for i in range(10):
    loss, acc = discriminator.train(X,y)
    print(loss, acc)

ypred = discriminator.predict(X).detach().numpy().flatten()

plt.figure(121)
plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.figure(122)
plt.scatter(X[ypred==0, 0], X[ypred==0, 1], label='predicted Class 0')
plt.scatter(X[ypred==1, 0], X[ypred==1, 1], label='predicted Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
