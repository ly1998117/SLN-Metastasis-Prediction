config = dict(
    num_classes=2,
    in_channels=1,
    pretrained=True,
    multi_label=False,
    select_labels=None,
    column_renames={'path_image': 'image',
                    'SLN状态（0_无转移，1_转移）': 'label',
                    'path_mask': 'mask'},
    postfix='',
)