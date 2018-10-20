# TODO: добавить типы
class Options:
    def __init__(self, n, ni, kn):
        self._number_points = n
        self._number_iter = ni
        self._k_noise = kn

    @property
    def number_points(self):
        return self._number_points

    @number_points.setter
    def number_points(self, val):
        self._number_points = val

    @property
    def ni(self):
        return self._number_iter

    @ni.setter
    def ni(self, val):
        self._number_iter = val

    @property
    def kn(self):
        return self._k_noise

    @kn.setter
    def kn(self, val):
        self._k_noise = val


class Algorithm:
    def __init__(self, class_name='', name='', full_name='', op=None, **kwargs):
        self._class_name = class_name
        self._name = name
        self._full_name = full_name

        self._options = op

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
