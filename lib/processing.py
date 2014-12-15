import scipy as sp
import numpy as np
from csv import DictReader
from collections import defaultdict
from sklearn.metrics import make_scorer


categorical = ['hour',
                'banner_pos',
                'site_id',
                'site_domain',
                'site_category',
                'app_id',
                'app_domain',
                'app_category',
                'device_id',
                'device_ip',
                'device_model',
                'device_type',
                'device_conn_type'
                ]

continuous = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']


def make_interact(iter):
    return ':'.join(iter)


def parse_row(row, categorical, continuous):   

    row['hour'] = str(int(row['hour']) % 24)

    for k in categorical:
        yield (row[k], 1.0)
        
        for j in categorical:
            if k != j:
                yield (make_interact([row[k],row[j]]), 1.0)

    yield (make_interact([row['app_category'], row['site_category'], row['banner_pos']]), 1.0)

    for i in continuous:
        yield (i, np.float(row[i]))
                
     
def get_field(field, path, generator=False):
    if not generator:
        return [int(row_i[field]) for row_i in DictReader(open(path))]
    else:
        return (int(row_i[field]) for row_i in DictReader(open(path)))


def data_generator(parser, path):
    #TODO readd kwdargs
    return (parser(x)
            for x in DictReader(open(path)))


def write_submission(number, ids, preds):
    with open('submissions/submission{}.csv'.format(number), 'w') as f:
        f.write('id,click\n')
        for id_, prob in zip(ids, preds):
            f.write('%s,%f\n' % (id_, prob))





