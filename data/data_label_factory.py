def __H_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    labels = [1 if 'fi' in name else 0 for name in file_names]
    return list(zip(data_files, labels))


def __M_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    labels = [1 if 'newfi' in name else 0 for name in file_names]
    return list(zip(data_files, labels))


def __VF_labeler(data_files):
    file_names = [name.split('/')[-1] for name in data_files]
    labels = [1 if 'violence' in name else 0 for name in file_names]
    return list(zip(data_files, labels))


def label_factory(data_name):
    if data_name == 'H': return __H_labeler
    if data_name == 'M': return __M_labeler
    if data_name == 'VF': return __VF_labeler
    assert 0, "Bad data name: " + data_name
