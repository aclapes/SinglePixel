import numpy as np
import os
import time
from sklearn.preprocessing import LabelEncoder
import argparse

from keras.layers import LSTM, Dense, Bidirectional, Activation
from keras.optimizers import Adam
from keras import Sequential
from keras.callbacks import EarlyStopping

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

problems = ['direction','path', 'sd_su', 'handwave', 'setup', 'f_r_hw_sd_su']

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
        if mode == 'row':
            return L
        elif mode == 'col':
            return {field_name : L[:,i] for i,field_name in enumerate(fields)}
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
        default=100,
        help=
        'Num epochs (default: %(default)s)')

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
        default=256,
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

    args = parser.parse_args()

    annots = read_dataset_annotation(args.annotation_filepath, mode='col')
    files = [os.path.join(annot[0].replace("\\",'/'),annot[1])
             for annot in zip(annots['folder'], annots['file_name'])]

    train_test_split = annots['train_test_split']

    st_time = time.time()
    try:
        data = np.load('/tmp/singlepixel_dataset.npy')
        print 'READING NPY took ', time.time() - st_time, ' secs.'
    except:
        data = read_dataset(args.input_dir, files)
        data = pad_temporal(data)
        print 'READING SEQUENCES (txt) took ', time.time() - st_time, ' secs.'
        np.save('/tmp/singlepixel_dataset.npy', data)

    for prob in problems:  # iterate over the different problems
        # find valid examples
        inds_train = np.where((annots[prob] != 'none') & (train_test_split == '0'))[0]
        inds_test = np.where((annots[prob] != 'none') & (train_test_split == '1'))[0]
        labels_train = annots[prob][inds_train]
        labels_test = annots[prob][inds_test]

        # encode labels to categorical and transform them to one-hot
        le = LabelEncoder()
        y_train = le.fit(labels_train) # encode categories
        y_train = categorical_to_onehot(le.fit_transform(labels_train), len(le.classes_))
        y_test = categorical_to_onehot(le.fit_transform(labels_test), len(le.classes_))

        # define keras model
        model = Sequential()

        if args.lstm_variation == 'onelayer_lstm':
            model.add(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2, input_shape=data.shape[1:]))
        elif args.lstm_variation == 'twolayer_lstm':
            model.add(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=data.shape[1:]))
            model.add(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2))
        elif args.lstm_variation == 'onelayer_bilstm':
            model.add(Bidirectional(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2),
                                    input_shape=data.shape[1:]))
        elif args.lstm_variation == 'twolayer_bilstm':
            model.add(Bidirectional(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
                                    input_shape=data.shape[1:]))
            model.add(Bidirectional(LSTM(args.hidden_size, dropout=0.2, recurrent_dropout=0.2)))
        else:
            raise NotImplementedError

        model.add(Dense(len(le.classes_)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        # model.summary()

        # run train and test
        history = model.fit(data[inds_train],
                            y_train,
                            # validation_data=(data[inds_test], y_test),
                            batch_size=args.batch_size,
                            epochs=args.num_epochs,
                            shuffle=True,
                            # callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, verbose=1)],
                            verbose=0)

        _, te_acc = model.evaluate(data[inds_test], y_test, batch_size=1, verbose=0)
        print('[Problem] %s : %.4f, %.4f' % (prob, history.history['acc'][-1], te_acc)
        )

    quit()
