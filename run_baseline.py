import numpy as np
import os
import time
from sklearn.preprocessing import LabelEncoder
import argparse

from keras.models import load_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from keras import backend as K

import pickle

from models import *
from utils import *
from reader import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The data and additional descriptions \
                    can be found at: https://github.com/aclapes/SinglePixel.'
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
        '-F',
        '--train-final',
        dest='train_final',
        action='store_true',
        help=
        'Train a final model with all data available'
    )

    parser.add_argument(
        '-E',
        '--evaluate-final',
        type=str,
        dest='evaluate_final',
        help=
        'Evaluate a final model with all data available'
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

    st_time = time.time()
    try:
        data_all = np.load('/tmp/singlepix_data.npy')
        data_center = np.load('/tmp/singlepix_center.npy')
        print 'READING NPY took ', time.time() - st_time, ' secs.'
    except:
        data_all = read_dataset(args.input_dir, files)
        data_all, data_center = pad_temporal(data_all)
        print data_center
        print 'READING SEQUENCES (txt) took ', time.time() - st_time, ' secs.'
        np.save('/tmp/singlepixtmp.npy', data_all)
        np.save('/tmp/singlepix_center.npy', data_center)

    # Iterate over the problems
    for prob in problems:
        # mask data for this particular problem (discard 'none' examples)
        mask_prob = annots[prob] != 'none'

        data = data_all[mask_prob]

        # encode labels
        le_classes = LabelEncoder()
        y = le_classes.fit_transform(annots[prob][mask_prob])
        classes = le_classes.classes_
        y_onehot = categorical_to_onehot(y, len(classes))
        print('[Problem] %s : #instances=%d, #classes=%d' % (prob, np.count_nonzero(mask_prob), len(classes)))

        model_name = '-'.join([args.lstm_variation, str(args.hidden_size), prob])
        if args.train_final:
            try:
                model = load_model(model_name + '.h5')
            except:
                model = init_model(args.lstm_variation, data.shape[1:], args.hidden_size, len(classes))
                labels_f, label_counts_f = np.unique(y, return_counts=True)
                model.fit(
                    data,
                    y_onehot,
                    batch_size=args.batch_size,
                    epochs=args.num_epochs,
                    class_weight={l: w for l, w in zip(labels_f, float(np.max(label_counts_f)) / label_counts_f)},
                    shuffle=True,
                    callbacks=[LearningRateScheduler(scheduler, verbose=0)],
                    verbose=1)
                model.save(model_name + '.h5')
        else:
            model = init_model(args.lstm_variation, data.shape[1:], args.hidden_size, len(classes))

            # split data for validation
            if args.k_folds > 0:
                skf = StratifiedKFold(n_splits=args.k_folds, random_state=42, shuffle=True)
                split = [s for s in skf.split(data, y)]
                n_splits = args.k_folds
            else:
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

            # mantain tr/te acc across folds
            acc_train = np.empty((n_splits,), dtype=np.float32)
            acc_test = np.empty_like(acc_train)
            conf_mat = np.zeros((n_splits, len(classes), len(classes)), dtype=np.int32)

            test_preds = []
            test_trues = []
            test_softmax = []

            init_weights = model.get_weights()
            for it, (train_inds, test_inds) in enumerate(split):
                # run train
                labels_tr, label_counts_tr = np.unique(y[train_inds], return_counts=True)
                class_weight = {l:w for l,w in zip(labels_tr, float(np.max(label_counts_tr)) / label_counts_tr)}

                callbacks = [LearningRateScheduler(scheduler, verbose=0)]
                if args.early_stop:
                    callbacks.append(EarlyStopping(monitor='val_loss', patience=8, verbose=0))
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

                acc_train[it] = history.history['acc'][-1]
                acc_test[it]  = te_acc_fold
                conf_mat[it, :, :] = confusion_matrix(y[test_inds], y_pred, labels=le_classes.transform(classes))

                test_preds.append(y_pred)
                test_trues.append(y[test_inds])
                test_softmax.append(y_softmax)

                model.set_weights(init_weights)

            with open(model_name + ('.logocv' if args.k_folds==0 else '.kfoldcv') + '.pkl', 'w') as pkl:
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

            print('[Problem-final] %s : %.4f, %.4f (weighted)' % (prob, acc_train/n_splits, acc_test/n_splits))
            print(le_classes.classes_)
            print np.sum(conf_mat, axis=0)

        if args.evaluate_final:
            annots_eval = read_dataset_annotation(args.evaluate_final, fields=problems, group=False)
            files_eval = [os.path.join(annot[0].replace("\\", '/'), annot[1])
                     for annot in zip(annots_eval['folder'], annots_eval['file_name'])]
            data_eval = read_dataset(args.input_dir, files_eval)
            data_eval = pad_temporal(data_eval, length=model.input_shape[1], center=data_center)

            # Iterate
            # mask data for this particular problem (discard 'none' examples)
            mask_prob = annots_eval[prob] != 'none'

            # run test
            y_softmax = model.predict(data_eval[mask_prob],
                                      batch_size=1,
                                      verbose=0)

            # Evaluation (weighted accuracy, ...)
            y = le_classes.transform(annots_eval[prob][mask_prob])
            y_onehot = categorical_to_onehot(y, len(classes))

            # compute class-normalized weights
            labels_eval, label_counts_eval = np.unique(y, return_counts=True)
            class_counts = {l: c for l, c in zip(labels_eval, label_counts_eval)}
            sample_weight = np.array([1. / class_counts[yi] for yi in y])
            sample_weight = sample_weight / np.sum(sample_weight)
            # binarize softmax outputs
            y_pred = np.argmax(y_softmax, axis=1)  # onehot to categorical
            y_onehot_pred = categorical_to_onehot(y_pred, y_softmax.shape[1])  # back to onehot
            # find hits and apply weights
            acc = np.sum(np.sum(y_onehot * y_onehot_pred, axis=1) * sample_weight)
            print('[Problem-feval] %s : %.4f' % (prob, acc))
            conf_mat = confusion_matrix(y, y_pred, labels=le_classes.transform(classes))

            with open(args.evaluate_final + '.' + model_name + '.feval.pkl', 'w') as pkl:
                dump_content = dict(
                    args=args,
                    folders=annots_eval['folder'][mask_prob],
                    file_names=annots_eval['file_name'][mask_prob],
                    annots_prob=annots_eval[prob][mask_prob],
                    classes=classes,
                    trues=y,
                    softmax=y_softmax,
                    acc=acc,
                    conf_mat=conf_mat
                )
                pickle.dump(dump_content, pkl)

    quit()
