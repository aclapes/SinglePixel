import numpy as np
from os.path import join

def read_dataset(parent_dir, files, pad=False):
    dataset = []

    for file in files:
        filepath = join(parent_dir, file)
        with open(filepath, 'r') as f:
            lines = []
            f.next() # discard first line
            for i in range(500):
                line = f.next().strip().split('\t')
                lines.append(line)
        D = np.array(lines, dtype=np.float32).T
        dataset.append(D)

    return dataset


def read_dataset_annotation(filepath, fields=None, group=True):
    """
    Reads dataset annotation one_pixel_data.csv, either row- or column-wise.
    :param filepath: path of one_pixel_data.csv file.
    :param mode: 'row' or 'col'
    :return: read data either row or column-wise.
        (if column-wise, it is a dictionary of pairs (field_name, 1-D field_data)
    """
    with open(filepath, 'r') as f:
        header = f.next().strip().split(',')
        lines = [[value.strip() for value in line.strip().split(',')] for line in f]

        L = np.array(lines)
        L_dict = {field: L[:,i] for i, field in enumerate(header)}
        if group:
            L_dict['group_id'] = np.array([file_name.split('_')[1] for file_name in L_dict['file_name']])

        for field in fields:
            if field not in header: # then we assume it's a compound field
                field_split = field.split('+')
                assert field_split[0] in header
                assert len(field_split) > 1  # not a composite field
                compound_field = L_dict[field_split[0]]
                for subfield in field_split[1:]:
                    assert subfield in header
                    compound_field = [(value + L_dict[subfield][i] if value != 'none' else 'none')
                                      for i,value in enumerate(compound_field)]
                L_dict[field] = np.array(compound_field)

        if fields:
            extra_fields = ['folder', 'file_name']
            if group:
                extra_fields.append('group_id')
            return {
                field_name : L_dict[field_name]
                for i,field_name in enumerate(fields + extra_fields)
            }
        else:
            return L_dict