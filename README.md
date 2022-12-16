# Normalized-SGD

In here, we introduce a new optimization algorithm used in machine learning models training which we call Normalized Stochastic Gradient Descent (NSGD) inspired by Normalized Least Mean Squares (NLMS) used in adaptive filtering. In large data sets, with high complexity models, choosing the learning rate is significant and the poor choice of parameters can lead to divergence. The algorithm updates the new set of network weights using the stochastic gradient but with  $\ell_1$ and $\ell_2$-based normalizations on the learning rate parameter similar to the NLMS algorithm. Our main difference from the existing normalization methods is that we do not include the error term in the normalization process. We experimentally observe that our optimization algorithm can train the model to a better accuracy level on different initial settings. In this paper, we demonstrate the efficiency of our training algorithm using ResNet18 and a toy neural network on different benchmark datasets with different initializations. The NSGD improves the accuracy of the ResNet-18 from 91.96\% to 92.20\% on CIFAR-10 dataset.
