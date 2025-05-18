def get_basic_config(config_name):
    if config_name == 'default':
        from .default import config
        return config
    elif config_name == 'multi':
        from .multiclass import config
        return config
    elif config_name == 's2':
        from .stage2 import config
        return config
    elif config_name == 'select':
        from .select import config
        return config
    else:
        raise ValueError('Unknown config name: {}'.format(config_name))
