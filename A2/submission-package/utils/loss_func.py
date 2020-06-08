import numpy as np


def linear_svm(W, b, x, y):
    """Linear SVM loss function.

    Parameters
    ----------
    W : ndarray
        The weight parameters of the linear classifier. D x C, where C is the
        number of classes, and D is the dimension of input data.

    b : ndarray
        The bias parameters of the linear classifier. C, where C is the number
        of classes.

    x : ndarray
        Input data that we want to predict the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    loss : float
        The average loss coming from this model. In the lecture slides,
        represented as \frac{1}{N}\sum_i L_i.
    """

    #compute matrix multiplication to get N x C ndarray
    scores = np.matmul(x, W) + b

    correct_scores = np.reshape( (scores[np.arange(len(y)), y]), (-1,1) )

    #calculate separate parts of svm
    loss_c = np.maximum(0, scores - correct_scores + 1)

    loss = np.mean(np.sum(loss_c, axis=1) - 1, axis=0)

    return loss


def logistic_regression(W, b, x, y):
    """Logistic regression loss function.

    Parameters
    ----------
    W : ndarray
        The weight parameters of the linear classifier. D x C, where C is the
        number of classes, and D is the dimension of input data.

    b : ndarray
        The bias parameters of the linear classifier. C, where C is the number
        of classes.

    x : ndarray
        Input data that we want to predict the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    loss : float
        The average loss coming from this model. In the lecture slides,
        represented as \frac{1}{N}\sum_i L_i.

    """
    #compute matrix multiplication to get N x C ndarray
    scores = np.matmul(x, W) + b
    scores = scores - np.max(scores)

    #obtain correct scores
    correct_scores = scores[np.arange(len(y)), y]

    #calculate separate parts of softmax
    e = np.exp(correct_scores)

    soft_loss = e / np.sum(np.exp(scores), axis=1, keepdims=True)

    log_soft_loss = (-1) * np.log(soft_loss[range(len(y)), y])
    loss = np.mean(log_soft_loss)


    return loss
