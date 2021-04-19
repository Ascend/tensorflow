class OptionValue(object):
    def __init__(self, default, optional):
        self.__default = default
        self.__optional = optional
        self.__value = default

    @property
    def default(self):
        return self.__default

    @property
    def optional(self):
        return self.__optional

    @property
    def value(self):
        if self.__value is None:
            return None
        if str(self.__value) == str(True):
            return "1"
        elif str(self.__value) == str(False):
            return "0"
        else:
            return str(self.__value)

    @value.setter
    def value(self, v):
        if isinstance(self.__optional, (tuple, list,)) and v not in self.__optional:
            raise ValueError("'" + str(v) + "' not in optional list " + str(self.__optional))
        self.__value = v


class NpuBaseConfig(object):

    def __init__(self):
        self._fixed_attrs = []
        for k, v in self.__dict__.items():
            if isinstance(v, (OptionValue, NpuBaseConfig)):
                self._fixed_attrs.append(k)

    def __setattr__(self, key, value):
        if hasattr(self, '_fixed_attrs'):
            if key not in self._fixed_attrs:
                raise ValueError(self.__class__.__name__ + " has no option " + key + ", all options " +
                                 str(self._fixed_attrs))
            if isinstance(getattr(self, key), OptionValue):
                getattr(self, key).value = value
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def as_dict(self):
        options = dict()
        for k, v in self.__dict__.items():
            if k in self._fixed_attrs:
                if isinstance(v, OptionValue) and v.value is not None:
                    options.update({k: v.value})
                elif isinstance(v, NpuBaseConfig):
                    options.update(v.as_dict())
        return options
