from .dataloader import DataHandler


def get_handler_from_args(args):
    data_handler = DataHandler(filepath=args.filepath,
                               column_renames=args.column_renames,
                               no_test=args.no_test,
                               postfix=args.postfix,
                               multi_label=args.multi_label,
                               select_labels=args.select_labels,
                               n_fold=args.n_fold,
                               n_repeat=args.n_repeat
                               )
    return data_handler


def get_dataset_from_args(args):
    data_handler = DataHandler(filepath=args.filepath,
                               column_renames=args.column_renames,
                               no_test=args.no_test,
                               postfix=args.postfix,
                               multi_label=args.multi_label,
                               select_labels=args.select_labels,
                               n_fold=args.n_fold,
                               n_repeat=args.n_repeat
                               )
    dataloader = data_handler.get_dataloader(repeat=args.repeat,
                                             fold=args.fold,
                                             batch_size=args.batch_size,
                                             workers=args.workers,
                                             unbalanced=args.unbalanced,
                                             spatial_size=args.spatial_size,
                                             norm=args.norm)
    return dataloader
