import numpy as np
import os
import time
from sklearn.preprocessing import LabelEncoder
import argparse

from keras.layers import LSTM, GRU, Dense, Bidirectional, Activation, Masking
from keras.optimizers import Adam
from keras import Sequential
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from keras import backend as K

def read_dataset_annotation(filepath, mode='row'):
    """
    Reads dataset annotation one_pixel_data.csv, either row- or column-wise.
    :param filepath: path of one_pixel_data.csv file.
    :param mode: 'row' or 'col'
    :return: read data either row or column-wise.
        (if column-wise, it is a dictionary of pairs (field_name, 1-D field_data)
    """

    fields = ['folder',
              'file_name',
              'repetition',
              'path',
              'direction',
              'sd_su',
              'handwave',
              'setup',
              'windows',
              'exceptions',
              'f_r_hw_sd_su',
              'train_test_split']

    with open(filepath, 'r') as f:

        header = f.next().strip().split(',')
        assert header == fields
        lines = [[value.strip() for value in line.strip().split(',')] for line in f]

        L = np.array(lines)

        # add derived attributes
        directed_path_col = np.array([(path + direction if path != 'none' else 'none')  for path, direction in zip(L[:,3], L[:,4])])
        L = np.concatenate([L, directed_path_col[:,np.newaxis]], axis=1)
        fields.append('directed_path')

        if mode == 'row':
            return L, fields
        elif mode == 'col':
            L_dict = {field_name : L[:,i] for i,field_name in enumerate(fields)}
            return L_dict
        else:
            return NotImplementedError


def read_dataset(parent_dir, files, pad=False):
    dataset = []

    for file in files:
        filepath = os.path.join(parent_dir, file)
        with open(filepath, 'r') as f:
            lines = []
            f.next() # discard first line
            for i in range(500):
                line = f.next().strip().split('\t')
                lines.append(line)
        D = np.array(lines, dtype=np.float32).T
        dataset.append(D)

    return dataset


def pad_temporal(X):
    """
    Pads temporal dimension (assuming it goes along 0-th axis) by adding zeros to beginning of the sequence.
    :param X: a list of TxF matrices. T are timesteps and F are features.
    :return: A 3-D array of matrices right padded along 0-th dimension with zeros.
    """
    max_len = np.max([X_i.shape[0] for X_i in X])
    mean_center = np.mean(np.concatenate(X, axis=0), axis=0)

    X_padded_norm = np.empty((len(X), max_len, 500), dtype=X[0].dtype)
    for i,X_i in enumerate(X):
        Xc = X_i - mean_center
        # <--
        X_padded_norm[i, -Xc.shape[0]:, :] = Xc  # more efficient
        # ---
        # pad_range = ((max(0, max_len - Xc.shape[0]),0), (0, 0))
        # X_padded_norm[i, :, :] = np.pad(Xc, pad_range, 'constant', constant_values=(0,0))
        # -->

    return X_padded_norm


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
        conf_mat[labels_lut[true[i]], labels_lut[pred[i]]] += 1

    return conf_mat


def scheduler(iteration):
    if iteration < 10:
        return 1e-3
    elif iteration < 20:
        return 5e-4
    else:
        return 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input-dir',
        type=str,
        dest='input_dir',
        default='./data/',
        help=
        'Dataset directory location (default: %(default)s)')

    parser.add_argument(
        '-a',
        '--annotation-filepath',
        type=str,
        dest='annotation_filepath',
        default='./one_pixel_data_v2.csv',
        help=
        'Annotation file location (default: %(default)s)')

    parser.add_argument(
        '-e',
        '--num_epochs',
        type=int,
        dest='num_epochs',
        default=50,
        help=
        'Num epochs (default: %(default)s)')

    parser.add_argument(
        '-r',
        '--early-stop',
        dest='early_stop',
        action='store_true',
        help=
        'Early stop flag (default: %(default)s)')

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        dest='batch_size',
        default=32,
        help=
        'Batch size (default: %(default)s)')

    parser.add_argument(
        '-l',
        '--lstm-variation',
        type=str,
        dest='lstm_variation',
        default='onelayer_lstm',
        help=
        'Choose between onelayer_lstm, twolayer_lstm, onelayer_bilstm, twolayer_bilstm (default: %(default))')

    parser.add_argument(
        '-s',
        '--hidden-size',
        type=int,
        dest='hidden_size',
        default=128,
        help=
        'Hidden size (default: %(default)s)')

    # parser.add_argument(
    #     '-G',
    #     '--gpu-memory',
    #     type=float,
    #     dest='gpu_memory',
    #     default=0.24,
    #     help=
    #     'GPU memory to reserve (default: %(default)s)')

    parser.add_argument(
        '-D',
        '--cuda-devices',
        type=str,
        dest='cuda_devices',
        default="3",
        help=
        'GPU devices (default: %(default)s)')

    parser.add_argument(
        '-k',
        '--k-folds',
        type=int,
        dest='k_folds',
        default=10,
        help=
        'Number of cross-validation folds (default: %(default)s)')

    parser.add_argument(
        '-P',
        '--problems',
        type=str,
        dest='problems',
        default='direction,path,directed_path,sd_su,handwave,setup,f_r_hw_sd_su',
        help=
        'List of problems to consider (default: %(default)s)')

    args = parser.parse_args()

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = args.cuda_devices
        set_session(tf.Session(config=config))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    annots = read_dataset_annotation(args.annotation_filepath, mode='col')
    files = [os.path.join(annot[0].replace("\\",'/'),annot[1])
             for annot in zip(annots['folder'], annots['file_name'])]

    train_test_split = annots['train_test_split']

    st_time = time.time()
    try:
        data_all = np.load('/tmp/singlepixel_dataset.npy')
        print 'READING NPY took ', time.time() - st_time, ' secs.'
    except:
        data_all = read_dataset(args.input_dir, files)
        data_all = pad_temporal(data_all)
        print 'READING SEQUENCES (txt) took ', time.time() - st_time, ' secs.'
        np.save('/tmp/singlepixel_dataset.npy', data_all)

    # Prepare splits for the different problems
    problems = [name.strip() for name in args.problems.split(',')]

    # Iterate
    for prob in problems:
        # mask data for this particular problem (discard 'none' examples)
        mask_prob = annots[prob] != 'none'

        # encode labels
        le = LabelEncoder()
        y = le.fit_transform(annots[prob][mask_prob])
        classes = le.classes_
        y_onehot = categorical_to_onehot(y, len(classes))
        print('[Problem] %s : #instances=%d, #classes=%d' % (prob, np.count_nonzero(mask_prob), len(classes)))

        # split data for validation
        skf = StratifiedKFold(n_splits=args.k_folds, random_state=42, shuffle=True)

        # mantain tr/te acc across folds
        acc_train = acc_test = 0.
        conf_mat = np.zeros((len(classes), len(classes)), dtype=np.int32)

        data = data_all[mask_prob,:]
        for train_inds, test_inds in skf.split(data, y):
            # Model definition
            model = Sequential()
            model.add(Masking(mask_value=0., input_shape=data.shape[1:]))

            # (lstm)
            if args.lstm_variation == 'onelayer_lstm':
                model.add(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2))
            elif args.lstm_variation == 'twolayer_lstm':
                model.add(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
                model.add(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2))
            elif args.lstm_variation == 'onelayer_bilstm':

                model.add(Bidirectional(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2)))
            elif args.lstm_variation == 'twolayer_bilstm':
                model.add(Bidirectional(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
                model.add(Bidirectional(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2)))
            # (gru)
            elif args.lstm_variation == 'onelayer_gru':
                model.add(GRU(args.hidden_size, dropout=0.2, recurrent_dropout=0.2))
            elif args.lstm_variation == 'twolayer_gru':
                model.add(GRU(args.hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
                model.add(GRU(args.hidden_size, dropout=0.2, recurrent_dropout=0.2))
            elif args.lstm_variation == 'onelayer_bigru':
                model.add(Bidirectional(GRU(args.hidden_size, dropout=0.2, recurrent_dropout=0.2)))
            elif args.lstm_variation == 'twolayer_bigru':
                model.add(Bidirectional(GRU(args.hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
                model.add(Bidirectional(GRU(args.hidden_size, dropout=0.2, recurrent_dropout=0.2)))
            else:
                raise NotImplementedError

            model.add(Dense(len(le.classes_)))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary()

            # Train and test

            # run train
            callbacks = [LearningRateScheduler(scheduler, verbose=0)]
            if args.early_stop:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=8, verbose=0))

            labels_tr, label_counts_tr = np.unique(y[train_inds], return_counts=True)
            class_weight = {l:w for l,w in zip(labels_tr, float(np.max(label_counts_tr)) / label_counts_tr)}

            history = model.fit(data[train_inds],
                                y_onehot[train_inds,:],
                                validation_data=(data[test_inds], y_onehot[test_inds,:]),
                                batch_size=args.batch_size,
                                epochs=args.num_epochs,
                                class_weight=class_weight,
                                shuffle=True,
                                callbacks=callbacks,
                                verbose=1)

            # run test
            y_softmax = model.predict(data[test_inds],
                                      batch_size=1,
                                      verbose=0)

            # Evaluation (weighted accuracy, ...)

            # compute class-normalized weights
            labels_te, label_counts_te = np.unique(y[test_inds], return_counts=True)
            class_counts = {l:c for l,c in zip(labels_te, label_counts_te)}
            sample_weight = np.array([1./class_counts[yi] for yi in y[test_inds]])
            sample_weight = sample_weight / np.sum(sample_weight)
            # binarize softmax outputs
            y_pred = np.argmax(y_softmax, axis=1)  # onehot to categorical
            y_onehot_pred = categorical_to_onehot(y_pred, y_softmax.shape[1]) # back to onehot
            # find hits and apply weights
            te_acc_fold = np.sum(np.sum(y_onehot[test_inds,:] * y_onehot_pred, axis=1) * sample_weight)
            print('[Problem-fold] %s : %.4f, %.4f' % (prob, history.history['acc'][-1], te_acc_fold)
            )

            conf_mat_fold = confusion_matrix(y[test_inds], y_pred, labels=le.transform(classes))

            acc_train += history.history['acc'][-1]
            acc_test  += te_acc_fold
            conf_mat  += conf_mat_fold

        print('[Problem-final] %s : %.4f, %.4f (weighted)' % (prob, acc_train/args.k_folds, acc_test/args.k_folds))
        print(le.classes_)
        print conf_mat

    quit()
