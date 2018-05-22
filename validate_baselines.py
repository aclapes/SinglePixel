import numpy as np
from os.path import join
import time
from sklearn.preprocessing import LabelEncoder
import argparse

from keras.models import load_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

import pickle

from models import *
from utils import *
from reader import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Single pixel action baseline implementations. '
                    'It implements LSTM and GRU temporal models and some variations of theirs.'
                    'These are validated in several problems. The data and additional descriptions'
                    'can be found at: https://github.com/aclapes/SinglePixel.'
    )

    parser.add_argument(
        '-i',
        '--input-dir',
        type=str,
        dest='input_dir',
        default='./data/',
        help=
        'Dataset directory location'
    )

    parser.add_argument(
        '-a',
        '--annotation-filepath',
        type=str,
        dest='annotation_filepath',
        default='./one_pixel_data_v3.csv',
        help=
        'Annotation file location'
    )

    parser.add_argument(
        '-l',
        '--lstm-variation',
        type=str,
        dest='lstm_variation',
        default='onelayer_lstm',
        help=
        'Choose between onelayer_lstm, twolayer_lstm, onelayer_bilstm, twolayer_bilstm, \
        onelayer_gru, twolayer_gru, onelayer_bigru, twolayer_bigru',
    )

    parser.add_argument(
        '-P',
        '--problems',
        type=str,
        dest='problems',
        default='forward,reverse,sd,su,handwave',
        help=
        'List of problems to consider (columns in .csv provided with -a option)'
    )

    parser.add_argument(
        '-s',
        '--hidden-size',
        type=int,
        dest='hidden_size',
        default=128,
        help=
        'LSTM/GRPU hidden layer size'
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        dest='batch_size',
        default=32,
        help=
        'Batch size'
    )

    parser.add_argument(
        '-e',
        '--num_epochs',
        type=int,
        dest='num_epochs',
        default=50,
        help=
        'Num epochs'
    )

    parser.add_argument(
        '-r',
        '--early-stop',
        dest='early_stop',
        action='store_true',
        help=
        'Early stop flag. If validation loss stopped decreasing'
    )

    parser.add_argument(
        '-k',
        '--k-folds',
        type=int,
        dest='k_folds',
        default=10,
        help=
        'Number of cross-validation folds (by default 10). If 0, leave-one-group-out is performed'
    )

    parser.add_argument(
        '-D',
        '--cuda-devices',
        type=str,
        dest='cuda_devices',
        default="0",
        help=
        'Comma-separated list of GPU devices. GPU with id=0 is used'
    )

    args = parser.parse_args()

    set_cuda_devices(args.cuda_devices)

    # Read data
    annots = read_dataset_annotation(args.annotation_filepath, fields=args.problems.split(','))
    files = [join(folder,file_name) for folder, file_name in zip(annots['folder'], annots['file_name'])]

    # Set some variable for validation procedure
    val_type = 'logocv' if args.k_folds == 0 else str(args.k_folds) + 'cv'
    callbacks = [LearningRateScheduler(scheduler, verbose=0)]
    if args.early_stop:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=8, verbose=0))

    st_time = time.time()
    try:
        data_all = np.load('/tmp/singlepix_data.npy')
        data_center = np.load('/tmp/singlepix_center.npy')
        print 'READING DATA (npy) took ', time.time() - st_time, ' secs.'
    except:
        data_all = read_dataset(args.input_dir, files)
        data_all, data_center = pad_temporal(data_all)
        print 'READING DATA (txt) took ', time.time() - st_time, ' secs.'
        np.save('/tmp/singlepixtmp.npy', data_all)
        np.save('/tmp/singlepix_center.npy', data_center)

    # Iterate over the problems
    for prob in args.problems.split(','):
        # mask data for this particular problem (discard 'none' examples)
        mask_prob = annots[prob] != 'none'

        data = data_all[mask_prob]

        # encode labels
        le_classes = LabelEncoder()
        y = le_classes.fit_transform(annots[prob][mask_prob])
        classes = le_classes.classes_
        y_onehot = categorical_to_onehot(y, len(classes))
        print('[Problem] %s : #instances=%d, #classes=%d' % (prob, np.count_nonzero(mask_prob), len(classes)))

        # Define model name and variables
        model_name = '-'.join([args.lstm_variation, str(args.hidden_size), prob])
        model = weights_init = None

        # split data for validation
        if args.k_folds > 0:  # cross-validation
            skf = StratifiedKFold(n_splits=args.k_folds, random_state=42, shuffle=True)
            split = [s for s in skf.split(data, y)]
            n_splits = args.k_folds
        else: # leave one group out validation
            skf = LeaveOneGroupOut()
            le_groups = LabelEncoder()
            if 'none' in annots[prob]:
                groups = le_groups.fit_transform(annots['group_id'][mask_prob])
                split = [s for s in skf.split(data, y, groups)]
            else:
                groups = le_groups.fit_transform(annots['group_id'][y==1])
                skf = StratifiedKFold(n_splits=len(le_groups.classes_), random_state=42, shuffle=True)
                split = [(np.concatenate([s[0],r[0]]),np.concatenate([s[1],r[1]]))
                         for s,r in zip(skf.split(data[y==1], y[y==1], groups), skf.split(data[y==0], y[y==0]))]
            n_splits = len(le_groups.classes_)
            print('[Problem] %s : groups={%s}' % (prob, ','.join(le_groups.classes_)))

        # mantain tr acc, te acc, confusion matrices, true labels, predicted labels, and so on

        acc_train = np.empty((n_splits,), dtype=np.float32)
        acc_train[:] = np.nan
        acc_test = np.empty_like(acc_train)
        conf_mat = np.zeros((len(classes), len(classes), n_splits), dtype=np.int32)

        test_trues = []
        test_softmax = []

        for it, (train_inds, test_inds) in enumerate(split):
            # run train
            labels_tr, label_counts_tr = np.unique(y[train_inds], return_counts=True)
            class_weight = {l:w for l,w in zip(labels_tr, float(np.max(label_counts_tr)) / label_counts_tr)}

            model_filepath = join('validation_models/', model_name, '%s.%s.%d.h5' % (model_name, val_type, it))
            try:
                model = load_model(model_filepath)
            except:
                if model is None:
                    model = init_model(args.lstm_variation, data.shape[1:], args.hidden_size, len(classes))
                    weights_init = model.get_weights()  # they will be re-initialized after each fold

                history = model.fit(data[train_inds],
                                y_onehot[train_inds,:],
                                validation_data=(data[test_inds], y_onehot[test_inds,:]),
                                batch_size=args.batch_size,
                                epochs=args.num_epochs,
                                class_weight=class_weight,
                                shuffle=True,
                                callbacks=callbacks,
                                verbose=1)

                try: os.makedirs(os.path.dirname(model_filepath))
                except OSError,e: pass

                # model.save(model_filepath)

            # run test
            y_softmax = model.predict(data[test_inds],
                                      batch_size=1,
                                      verbose=0)

            # Evaluation (weighted accuracy, ...)
            te_acc_fold = compute_accuracy(y[test_inds], y_softmax)

            # compute class-normalized weights
            print('[Problem-%s.%d] %s : test_acc(w)=%.4f'
                  % (val_type, it, prob, te_acc_fold))

            acc_test[it]  = compute_accuracy(y[test_inds], y_softmax)
            conf_mat[:,:,it] = confusion_matrix(y[test_inds], np.argmax(y_softmax,axis=1), labels=le_classes.transform(classes))

            test_trues.append(y[test_inds])
            test_softmax.append(y_softmax)

            if weights_init:
                model.reset_states()
                model.set_weights(weights_init)

        results_filepath = join('results', '%s.%s.pkl' % (model_name, val_type))

        try: os.makedirs(os.path.dirname(results_filepath))
        except OSError,e: pass

        with open(results_filepath, 'w') as pkl:
            dump_content = dict(
                args=args,
                annots_prob=annots[prob][mask_prob],
                classes=classes,
                split=split,
                test_trues=test_trues,
                test_softmax=test_softmax,
                acc_test=acc_test,
                conf_mat=conf_mat
            )
            pickle.dump(dump_content, pkl)

        print(classes)
        conf_mat = np.sum(conf_mat, axis=-1)
        print conf_mat

        print('[Problem-final] %s : train_acc=%.4f, test_acc(w)=%.4f'
              % (prob, np.mean(acc_train), np.mean(acc_test)))