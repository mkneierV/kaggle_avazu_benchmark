from csv import DictReader


features = ['hour',
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
            'device_conn_type',
            'C1',
            'C15',
            'C16',
            'C20',
            'C14',
            'C15',
            'C17',
            'C19',
            'C21']


def _make_interact(iterable):
    return ':'.join(iterable)


def _clean_row(X):
    X['hour'] = str(int(X['hour']) % 24)


def clean_parse_row(row, features=features): 

    _clean_row(row)

    for k in features:
        yield (':'.join((k, row[k])), 1.0)
        
        for j in features:
            if k != j:
                yield (_make_interact([k, j, row[k],row[j]]), 1.0)

    yield (_make_interact([row['app_category'], row['site_category'], row['banner_pos']]), 1.0)


def get_int_field(field, path):
    with open(path) as f:
        for row in DictReader(f):
            yield (int(row[field]) if row[field] != '' else 0)


def data_generator(parser, path, rowfeatures=features):
    """
    Needs to return vanilla str, not numpy str to work with hashing
    """
    with open(path) as f:
        for row in DictReader(f):
            yield (parser(row, rowfeatures))


def write_submission(number, ids, preds):
    with open('submissions/submission{}.csv'.format(number), 'w') as f:
        f.write('id,click\n')
        for id_, prob in zip(ids, preds):
            f.write('%s,%f\n' % (id_, prob))
    f.close()

