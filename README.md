# Normalized-SGD

In here, we introduce a new optimization algorithm used in machine learning models training which we call Normalized Stochastic Gradient Descent (NSGD) inspired by Normalized Least Mean Squares (NLMS) used in adaptive filtering. In large data sets, with high complexity models, choosing the learning rate is significant and the poor choice of parameters can lead to divergence. The algorithm updates the new set of network weights using the stochastic gradient but with  $\ell_1$ and $\ell_2$-based normalizations on the learning rate parameter similar to the NLMS algorithm. Our main difference from the existing normalization methods is that we do not include the error term in the normalization process. We experimentally observe that our optimization algorithm can train the model to a better accuracy level on different initial settings. In this paper, we demonstrate the efficiency of our training algorithm using ResNet-20 and a toy neural network on different benchmark datasets with different initializations. The NSGD improves the accuracy of the ResNet-20 from 91.96\% to 92.20\% on CIFAR-10 dataset.

Let us assume that we have a linear neuron at the last stage of the network. 
let $e_n$ be the error due to the n-th training instance.
We update the weight vector $w_k$ of the k-th neuron of the last layer as follows

$w_{k+1} = w_k + \lambda \frac{e_n}{||x_n||^2_2}    x_n $

where $x_n$ is the input feature map vector of the k-th neuron due to the
$n$-th instance. This equation is essentially the same as the NLMS equation.

The figure showing how the optimizer algorithm works for any layer is shown in the figure "alg_fig.png". For any other layer beside the last layer, the formula becomes slightly different. We use the gradient term as it is and normalize it with the input. 

$w_k \leftarrow w_k + \lambda \frac{\nabla_{w(k)}e_n}{||x_n||^2_2} $


