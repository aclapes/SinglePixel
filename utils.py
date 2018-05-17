import numpy as np
from keras import backend as K
import os

def set_cuda_devices(devices_list):
    # Deal with visible GPU devices
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = devices_list
        set_session(tf.Session(config=config))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = devices_list


def pad_temporal(X, length=None, center=None):
    """
    Pads temporal dimension (assuming it goes along 0-th axis) by adding zeros to beginning of the sequence.
    :param X: a list of TxF matrices. T are timesteps and F are features.
    :return: A 3-D array of matrices right padded along 0-th dimension with zeros.
    """

    if length:
        max_len = length
    else:
        max_len = np.max([X_i.shape[0] for X_i in X])

    if center:
        mean_center = center
    else:
        mean_center = np.mean(np.concatenate(X, axis=0))

    X_padded_norm = np.empty((len(X), max_len, X[0].shape[1]), dtype=X[0].dtype)
    for i, X_i in enumerate(X):
        Xc = X_i - mean_center
        # <--
        X_padded_norm[i, -Xc.shape[0]:, :] = Xc  # more efficient
        # ---
        # pad_range = ((max(0, max_len - Xc.shape[0]),0), (0, 0))
        # X_padded_norm[i, :, :] = np.pad(Xc, pad_range, 'constant', constant_values=(0,0))
        # -->

    if center:
        return X_padded_norm
    else:
        return X_padded_norm, mean_center


def categorical_to_onehot(y, n_classes, dtype=np.int32):
    y_onehot = np.zeros((len(y), n_classes), dtype=dtype)
    y_onehot[range(len(y)),y] = 1  # categorical to one-hot

    return y_onehot


def confusion_matrix(true, pred, labels=None):
    assert len(true) == len(pred)
    labels_true = np.unique(true)

    if labels is None:
        labels = labels_true
    else:
        assert len(labels_true) <= len(labels)

    labels_lut = {lbl: i for i, lbl in enumerate(labels)}

    conf_mat = np.zeros((len(labels_lut), len(labels_lut)), dtype=np.int32)
    for i in range(len(true)):
        conf_mat[labels_lut[pred[i]],labels_lut[true[i]]] += 1

    return conf_mat

def compute_accuracy(y_true, y_softmax):
    labels_te, label_counts_te = np.unique(y_true, return_counts=True)
    class_counts = {l: c for l, c in zip(labels_te, label_counts_te)}
    sample_weight = np.array([1. / class_counts[yi] for yi in y_true])
    sample_weight = sample_weight / np.sum(sample_weight)

    # binarize softmax outputs
    y_onehot_true = categorical_to_onehot(y_true, y_softmax.shape[1])
    y_pred = np.argmax(y_softmax, axis=1)  # onehot to categorical
    y_onehot_pred = categorical_to_onehot(y_pred, y_softmax.shape[1])  # back to onehot

    # find hits and apply weights
    return np.sum(np.sum(y_onehot_true * y_onehot_pred, axis=1) * sample_weight)

def evaluation_metrics(confmat, metrics=['prec','recall','fscore']):
    diag = np.diag(confmat).astype('float32')
    fp = np.sum(confmat, axis=1) - diag
    fn = np.sum(confmat, axis=0) - diag
    prec = diag / (diag+fp)
    recall = diag / (diag+fn)
    computed_metrics = {
        'tp' : diag,
        'fp' : fp,
        'fn' : fn,
        'prec' : prec,
        'recall' : recall,
        'fscore' : 2./((1./prec) + (1./recall)),
    }
    return [computed_metrics[m] for m in metrics]