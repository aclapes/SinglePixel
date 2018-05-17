import numpy as np
import cPickle
from os.path import join
from collections import OrderedDict

# evaluations = ['One robot standing still',
#                'Simultaneous action',
#                 'Two different actions']
evaluations = ['Two different actions']
ext = '.csv'
problems = ['forward', 'reverse', 'sd', 'su', 'handwave']

model_name = 'twolayer_bigru-256'


for eval in evaluations:
    D = OrderedDict()
    stats = OrderedDict()
    conf_mats = OrderedDict()
    for prob in problems:
        results_file = eval + ext + '.' + model_name + '-' + prob + '.feval.pkl'
        with open(results_file) as f:
            content_pkl = cPickle.load(f)
            filepaths = zip(content_pkl['folders'], content_pkl['file_names'])
            for i, (true, softmax) in enumerate(zip(content_pkl['trues'], content_pkl['softmax'])):
                fp = join(content_pkl['folders'][i],content_pkl['file_names'][i])
                D.setdefault(fp,[]).append((true, np.argmax(softmax)))

            conf_mats[prob] = content_pkl['conf_mat'].astype('float32')
            prec = conf_mats[prob][0,0] / np.sum(conf_mats[prob][0,:])
            recall = conf_mats[prob][0,0] / np.sum(conf_mats[prob][:,0])
            fscore = 2./(1./prec + 1./recall)
            acc = (conf_mats[prob][0,0] + conf_mats[prob][1,1]) / np.sum(conf_mats[prob])
            stats[prob] = (prec, recall, fscore, acc)

    print eval + ':' + str(problems)
    for k, v in D.iteritems():
        print k + "\t" + str(v)

    print eval + ': confusion matrix'
    for k, v in conf_mats.iteritems():
        print k
        print v

    print eval + ': prec, recall, fscore, acc'
    for k, v in stats.iteritems():
        print k + "\t" + str(v)
