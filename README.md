# Normalized-SGD

In here, we introduce a new optimization algorithm used in machine learning models training which we call Normalized Stochastic Gradient Descent (NSGD) inspired by Normalized Least Mean Squares (NLMS) used in adaptive filtering. In large data sets, with high complexity models, choosing the learning rate is significant and the poor choice of parameters can lead to divergence. The algorithm updates the new set of network weights using the stochastic gradient but with  $\ell_1$ and $\ell_2$-based normalizations on the learning rate parameter similar to the NLMS algorithm. Our main difference from the existing normalization methods is that we do not include the error term in the normalization process. We experimentally observe that our optimization algorithm can train the model to a better accuracy level on different initial settings. In this paper, we demonstrate the efficiency of our training algorithm using ResNet18 and a toy neural network on different benchmark datasets with different initializations. The NSGD improves the accuracy of the ResNet-18 from 91.96\% to 92.20\% on CIFAR-10 dataset.

Let us assume that we have a linear neuron at the last stage of the network and %the cost function is the mean-squared error (MSE). 
let $e_n$ be the error due to the n-th training instance.
We update the weight vector $w_k$ of the k-th neuron of the last layer as follows
\begin{equation} \label{eq:NSGD1}
w_{k+1} = w_k + \lambda \frac{e_n}{||x_n||^2_2}    x_n 
\end{equation}
where $x_n$ is the input feature map vector of the k-th neuron due to the
$n$-th instance. This equation is essentially the same as the NLMS equation Eq. (3).

The figure showing how the optimizer algorithm works for any layer is shown in Fig. \ref{fig:backprop}. For any other layer beside the last layer, the formula becomes slightly different. We use the gradient term as it is and normalize it with the input. 
\begin{equation} \label{eq:NSGD2}
w_k \leftarrow w_k + \lambda \frac{\nabla_{w(k)}e_n}{||x_n||^2_2}
\end{equation}
Theoretically, (\ref{eq:NSGD1}) and (\ref{eq:NSGD2}) are the same if the cost function is mean squared error.
