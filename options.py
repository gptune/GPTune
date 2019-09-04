class Options(Mapping):

    def __init__(self, **kwargs):

        self.update(kwargs)

        self['parallel_tuning'] = False

