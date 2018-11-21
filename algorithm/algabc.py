# TODO: добавить типы
import support


class Options:
    _alias_map = {
        'number_points': ['n', 'np'],
        'number_iter': ['ni', 'iter'],
        'k_noise': ['kn']
    }
    _required_keys = ('number_points', 'number_iter')

    def __init__(self, **kwargs):
        kw = support.normalize_kwargs(kwargs,
                                      alias_map=self.__class__._alias_map,
                                      required=self.__class__._required_keys)
        self._number_points = kw['number_points']
        self._number_iter = kw['number_iter']
        self._k_noise = 0 if 'k_noise' not in kw else kw['k_noise']

    def update_op(self, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=self.__class__._alias_map)
        for k, v in kw.items():
            print(k, v)
            if k in self.__class__._alias_map:
                self.__setattr__(k, v)

    @property
    def number_points(self):
        return self._number_points

    @number_points.setter
    def number_points(self, val):
        self._number_points = val

    @property
    def number_iter(self):
        return self._number_iter

    @number_iter.setter
    def number_iter(self, val):
        self._number_iter = val

    @property
    def k_noise(self):
        return self._k_noise

    @k_noise.setter
    def k_noise(self, val):
        self._k_noise = val


class Algorithm:
    def __init__(self, class_name='', name='', full_name='', op=None, **kwargs):
        self._class_name = class_name
        self._name = name
        self._full_name = full_name

        self._options = op

    def probability_estimate(self, *args, **kwargs):
        pass

    @property
    def class_name(self):
        return self._class_name

    @property
    def name(self):
        return self._name

    @property
    def full_name(self):
        return self._full_name

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, val):
        self._options = val


class GSA(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(class_name='GSA', **kwargs)
        # self._class_name = 'GSA'


class SAC(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(class_name='SAC', **kwargs)
        # self._class_name = 'SAC'


class ASA(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(class_name='ASA', **kwargs)


class PSO(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(class_name='PSO', **kwargs)
