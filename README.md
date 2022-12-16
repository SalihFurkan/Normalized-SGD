# Normalized-SGD

In here, we introduce a new optimization algorithm used in machine learning models training which we call Normalized Stochastic Gradient Descent (NSGD) inspired by Normalized Least Mean Squares (NLMS) used in adaptive filtering. In large data sets, with high complexity models, choosing the learning rate is significant and the poor choice of parameters can lead to divergence. The algorithm updates the new set of network weights using the stochastic gradient but with  $\ell_1$ and $\ell_2$-based normalizations on the learning rate parameter similar to the NLMS algorithm.  
