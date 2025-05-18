from .default import config

config.update(dict(
    root_dir='run_s2',
    column_renames={'path_data': 'image',
                    'SLN转移个数（0_转移≤2枚，1_转移≥3枚)': 'label',
                    'path_mask': 'mask'},
    postfix='stage2',
    select_labels=None,
))
