# TODO: добавить типы
import itertools

import numpy as np

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
                                      alias_map=Options._alias_map,
                                      required=Options._required_keys)
        print(kw)
        self._number_points = kw['number_points']
        self._number_iter = kw['number_iter']
        self._k_noise = 0 if 'k_noise' not in kw else kw['k_noise']

    def update_op(self, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=Options._alias_map)
        for k, v in kw.items():
            print(k, v)
            if k in Options._alias_map:
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

    def probability_estimate(self, tf, op, iteration: dict, ep: float=0.2, number_runs: int=100, min_flag: int=1,
                             *args, **kwargs):
        ar = list(iteration.values())
        size = tuple(len(i) for i in ar)
        idxs = list(itertools.product(*(list(range(len(i))) for i in ar)))
        items = list((dict(zip(iteration.keys(), values)) for values in itertools.product(*iteration.values())))
        res = np.zeros(size)
        for i in range(len(idxs)):
            print('index:', idxs[i])
            print('item:', items[i])
            op.update_op(**items[i])
            print(op)
            p = 0
            for j in range(number_runs):
                x_bests, *_ = self.optimization(tf, min_flag=min_flag, *args, **kwargs)
                if tf.in_vicinity(x_bests, epsilon=ep):
                    p += 1
            res[idxs[i]] = p / number_runs
            print('Оценка вероятности', res[idxs[i]])
            print('-' * 20)
        return res

    def optimization(self, *args, **kwargs):
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


class SAC(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(class_name='SAC', **kwargs)


class ASA(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(class_name='ASA', **kwargs)


class PSO(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(class_name='PSO', **kwargs)
