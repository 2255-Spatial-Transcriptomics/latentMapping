import matplotlib.pyplot as plt 
from models.descriminatorModels.classifier_trainer import BinaryClassifierTrainer
import numpy as np
discriminator = BinaryClassifierTrainer(n_latent=10)

from sklearn.decomposition import PCA


### test 
# X0 = np.genfromtxt('data/latent_spaces/adipose_latent_250epoch.tsv')
# X1 = np.genfromtxt('data/latent_spaces/lung_human_ASK440_latent_250epoch.tsv')
X0 = np.genfromtxt('data/latent_spaces/scRNA_10D_shared_genes.tsv')
X1 = np.genfromtxt('data/latent_spaces/st_10D_shared_genes.tsv')
X = np.concatenate((X0, X1))
y = np.concatenate((np.zeros(X0.shape[0]), np.ones(X1.shape[0])))
    
pca = PCA(n_components=2)
pca.fit(X0)
X2d = pca.fit_transform(X)


### test for noisy 2D data
# num_samples = 5000
# mean_0 = np.array([0, 0])
# cov_0 = np.array([[1, 0.5], [0.5, 2]])  # covariance matrix for class 0
# mean_1 = np.array([2, 2])
# cov_1 = np.array([[1, -0.5], [-0.5, 1]])  # covariance matrix for class 1
# X_0 = np.random.multivariate_normal(mean_0, cov_0, num_samples // 2)
# X_1 = np.random.multivariate_normal(mean_1, cov_1, num_samples // 2)
# X = np.concatenate((X_0, X_1))
# y = np.concatenate((np.zeros(num_samples // 2), np.ones(num_samples // 2)))

for i in range(10):
    loss, acc = discriminator.train(X,y)
    print(loss, acc)

ypred = discriminator.predict(X).detach().numpy().flatten()

plt.figure(121)
plt.scatter(X2d[y==0,0], X2d[y==0,1], label='scRNA', s=1)
plt.scatter(X2d[y==1,0], X2d[y==1,1], label='ibST', s=1)
plt.title('ground truth')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.figure(122)
plt.scatter(X2d[ypred==0,0], X2d[ypred==0,1], label='predicted for scRNA',s=1)
plt.scatter(X2d[ypred==1,0], X2d[ypred==1,1], label='predicted for ibST',s=1)
plt.title(f'acc = {acc}')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()