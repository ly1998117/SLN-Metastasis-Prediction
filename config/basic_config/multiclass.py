from .default import config

config.update(dict(
    root_dir='run_multi',
    column_renames={'path_data': 'image',
                    'multi_label': 'label',
                    'path_mask': 'mask'},
    postfix='multi',
    num_classes=3,
    multi_label=True,
))
