import tensorflow as tf
import tensorflow.keras.backend as K


def jaccard_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the jaccard coefficient, a measure of similarity between two sets.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth labels.
    y_pred : tf.Tensor
        Predicted labels.

    Returns
    -------
    out : tf.Tensor
        jaccard coefficient for each sample in the batch.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jaccard = (intersection + 1e-15) / (sum_ - intersection + 1e-15)
    return jaccard
