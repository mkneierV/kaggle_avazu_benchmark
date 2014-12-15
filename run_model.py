import logging
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

import lib.ml as ml
import lib.preprocessing as pp


def main(neg_rate, submission_num, n_iter, train_path):
    ids = [x for x in pp.get_int_field('id', 'original_data/test')]
    clicks = pp.get_int_field('click', train_path)
    # Get Data Generators
    train = pp.data_generator(pp.clean_parse_row, train_path)
    test = pp.data_generator(pp.clean_parse_row, 'original_data/test')

    # Define estimators
    fh = FeatureHasher(n_features=2 ** 20, input_type='pair')
    sgd = SGDClassifier(loss='log', n_iter=1, alpha=.003, penalty='l2')

    #Fit pipeline
    pipeline = ml.PartialFitter([fh, sgd],
                                batch_size=10000,
                                logging=True,
                                n_iter=n_iter,
                                neg_rate=neg_rate)

    pipeline.partial_fit(X=train, y=clicks)
    # Correct Intercept
    pipeline.steps[-1].intercept_[0] += np.log(neg_rate)
    preds = pipeline.predict_proba(newX=test)[:, 1]
    pp.write_submission(number=submission_num, ids=ids, preds=preds)


if __name__ == '__main__':
    logging.basicConfig(filename='train_errors.log', level=logging.WARNING)
    from docopt import docopt
    import sys

    usage = '''Train SGD model and create entry for Kaggle Avazu Competition.
        Usage:
        %(program_name)s --neg_rate=<r> --submission_num=<s> --n_iter=<n> --train_path=<p>
        %(program_name)s (-h | --help)
        Options:
        -h --help           Show this screen.
        --neg_rate=<r>          Rate at which to sample negative cases
        --submission_num=<s>    Submission number.
        --n_iter=<n>            Number of fitting iterations over training data
        --train_path=<p>        Path to training dataset
        ''' % {'program_name': sys.argv[0]}

    arguments = docopt(usage)

    main(np.float(arguments['--neg_rate']),
         arguments['--submission_num'],
         np.int(arguments['--n_iter']),
         arguments['--train_path']
    )



