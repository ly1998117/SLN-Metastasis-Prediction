from .default import config

config.update(dict(
    root_dir='run_multi',
    column_renames={'path_data': 'image',
                    'multi_label': 'label',
                    'path_mask': 'mask'},
    postfix='select',
    num_classes=2,
    multi_label=True,
    select_labels=[1, 2],
))