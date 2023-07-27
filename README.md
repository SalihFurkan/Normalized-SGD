# Input Normalized-SGD

In this paper, we propose a novel optimization algorithm for training machine learning models called Input Normalized Stochastic Gradient Descent (INSGD), inspired by the Normalized Least Mean Squares (NLMS) algorithm used in adaptive filtering. When training complex models on large datasets, the choice of optimizer parameters, particularly the learning rate, is crucial to avoid divergence. Our algorithm updates the network weights using stochastic gradient descent with $\ell_1$ and $\ell_2$-based normalizations applied to the learning rate, similar to NLMS. However, unlike existing normalization methods, we exclude the error term from the normalization process and instead normalize the update term using the input vector to the neuron. Our experiments demonstrate that our optimization algorithm achieves higher accuracy levels compared to different initialization settings. We evaluate the efficiency of our training algorithm on benchmark datasets using ResNet-20, WResNet-18, ResNet-50, and a toy neural network. Our INSGD algorithm improves the accuracy of ResNet-20 on CIFAR-10 from 92.55\% to 92.80\%, the accuracy of MobileNetV3 on CIFAR-10 from 90.83\% to 91.13\%, WResNet-18 on CIFAR-100 from 78.75\% to 78.85\%, and ResNet-50 on ImageNet-1K from 75.56\% to 75.89\%.

Let us assume that we have a linear neuron at the last stage of the network. 
let $e_n$ be the error due to the n-th training instance.
We update the weight vector $w_k$ of the k-th neuron of the last layer as follows

$w_{k+1} = w_k - \lambda \frac{e_n}{||x_n||^2_2}    x_n $

where $x_n$ is the input feature map vector of the k-th neuron due to the
$n$-th instance. This equation is essentially the same as the NLMS equation.

The figure showing how the optimizer algorithm works for any layer is shown in the figure "alg_fig.png". For any other layer beside the last layer, the formula becomes slightly different. We use the gradient term as it is and normalize it with the input. 

$w_{k+1} = w_k - \mu \frac{\nabla_{w_k}L(e_k)}{\epsilon + ||x_k||^2_2}$

we incorporate momentum, a technique that aids in navigating high error and low curvature regions. In the INSGD algorithm, we introduce an input momentum term to estimate the power of the dataset, enabling power normalization. By replacing the denominator term with the estimated input power, we emphasize the significance of power estimation in our algorithm. Furthermore, the utilization of input momentum allows us to capture the norm of all the inputs. Denoted as $P$, the input momentum term accumulates the squared $\ell_2$ norm of the input instances:


