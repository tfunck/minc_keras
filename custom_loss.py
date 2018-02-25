from keras import backend as K
import numpy as np

def dice_loss(y_true, y_pred):
    """
    Computes approximate DICE coefficient as a loss by using the negative, computed with the Keras backend. The overlap\
    and total are offset to prevent 0/0, and the values are not rounded in order to keep the gradient information.
    Args:
    :arg y_true: Ground truth
    :arg y_pred: Predicted value for some input
    Returns
    :return: Approximate DICE coefficient.
    """
    ytf = K.flatten(y_true)
    ypf = K.flatten(y_pred)

    overlap = K.sum(ytf*ypf)
    total = K.sum(ytf*ytf) + K.sum(ypf * ypf)
    return -(2*overlap +1e-10) / (total + 1e-10)


def dice_metric(y_true, y_pred):
    """
    Computes DICE coefficient, computed with the Keras backend.
    Args:
    :arg y_true: Ground truth
    :arg y_pred: Predicted value for some input
    Returns
    :return: DICE coefficient
    """
    ytf = K.round(K.flatten(y_true))
    ypf = K.round(K.flatten(y_pred))

    overlap = 2*K.sum(ytf*ypf)
    total = K.sum(ytf*ytf) + K.sum(ypf * ypf)

    return overlap / total
