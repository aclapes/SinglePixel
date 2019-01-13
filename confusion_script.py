import numpy as np
import fileinput
import re
import sys
import os
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
np.set_printoptions(precision=2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()


for filename in fileinput.input():
    filename = filename.strip()
    with open(filename, 'r') as f:
        is_final = False
        classes = None
        problem_name = None
        row_list = []
        for line in f:
            if 'Problem-final' in line:
                is_final = True
                continue

            if is_final and classes is None:
                classes = re.sub('[^0-9A-Za-z\s]', '', line.strip()).split(' ')
            elif is_final:
                row = re.sub('\s+', ' ', re.sub('[^0-9\s]', '', line.strip()).strip()).split(' ')
                row_list.append(row)

        conf_mat = np.array(row_list, dtype=np.int32)
        print filename
        print classes
        print conf_mat
        plt.figure()
        plot_confusion_matrix(conf_mat, classes=classes, normalize=False, title=os.path.splitext(filename)[0])
        plt.savefig(filename + '.png')
        # with open(sys.argv[1] + '.conf_mat.npy', 'w') as pkl:
        #     pickle.dump({'classes' : classes, 'conf_mat' : conf_mat}, pkl)
