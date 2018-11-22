import operator
import numpy as np

import support
from algorithm.algabc import Options, PSO


class PSAOptions(Options):
    _alias_map = {
        'omega': ['o'],
        'fi_p': ['fp'],
        'fi_g': ['fg'],
    }
    _required_keys = ('omega', 'fi_p', 'fi_g')

    def __init__(self, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=PSAOptions._alias_map, required=PSAOptions._required_keys)
        super().__init__(**kw)
        self._omega = kw['omega']
        self._fi_p = kw['fi_p']
        self._fi_g = kw['fi_g']

    def update_op(self, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=PSAOptions._alias_map)
        for k, v in kw.items():
            print(k, v)
            if k in PSAOptions._alias_map:
                self.__setattr__(k, v)
        super().update_op(**kw)

    def __repr__(self):
        return (f'PSAOptions(number_points={self._number_points}, number_iter={self._number_iter}, '
                f'k_noise={self._k_noise}, omega={self._omega}, fi_p={self._fi_p}, fi_g={self._fi_g})')

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, v):
        self._omega = v

    @property
    def fi_p(self):
        return self._fi_p

    @fi_p.setter
    def fi_p(self, v):
        self._fi_p = v

    @property
    def fi_g(self):
        return self._fi_g

    @fi_g.setter
    def fi_g(self, v):
        self._fi_g = v


class Point:
    _alias_map = {'coordinates': ['c'], 'value': ['val'],
                  'velocity': ['vel'], 'best_coord': ['bc'], 'best_val': ['bv']}
    _required_keys = ('coordinates', )

    def __init__(self, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=Point._alias_map, required=Point._required_keys)
        self._coordinates = kw['coordinates'].copy()
        self._value = 0 if 'value' not in kw else kw['value']
        self._velocity = None if 'velocity' not in kw else kw['velocity'].copy()
        self._best_value = 0 if 'best_val' not in kw else kw['best_val']
        self._best_coordinates = kw['coordinates'].copy()

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, val):
        self._coordinates = val.copy()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, val):
        self._velocity = val

    @property
    def best_value(self):
        return self._best_value

    @best_value.setter
    def best_value(self, val):
        self._best_value = val

    @property
    def best_coord(self):
        return self._best_coordinates

    @best_coord.setter
    def best_coord(self, val):
        self._best_coordinates = val.copy()

    def __repr__(self):
        return (f'Point(coordinates={self._coordinates}, value={self._value}, '
                f'velocity={self._velocity}, best_coord={self._best_coordinates}, '
                f'best_val={self._best_value})')


class StandardPSA(PSO):
    def __init__(self, op, **kwargs):
        super().__init__(op=op, **kwargs)
        self._name = 'Standard PSA'
        self._full_name = 'Standard particle swarm algorithm'

    def optimization(self, tf, min_flag=1):
        if self._options:
            return psa(self._options, tf, min_flag)
        raise ValueError('Не установлены параметры алгоритма')


def psa(op, tf, min_flag):
    points = initialization(op, tf)
    best_point = get_best_point(points, min_flag)
    best_x, func_best = best_point.coordinates, best_point.value
    iteration = 0
    best_chart = np.zeros((op.number_iter,))
    mean_chart = np.zeros((op.number_iter,))
    for i in range(op.number_iter):
        iteration = i + 1
        best_chart[i] = func_best
        mean_chart[i] = np.mean([p.value for p in points])
        for p in points:
            r_p = np.random.uniform(0, 1, (tf.dim,))
            r_g = np.random.uniform(0, 1, (tf.dim,))
            p.velocity = op.omega * p.velocity + op.fi_p * r_p * (p.best_coord - p.coordinates) + op.fi_g * r_g * (best_x - p.coordinates)
            p.coordinates = p.coordinates + p.velocity
            p.value = tf.get_value(p.coordinates) + np.random.uniform(-tf.amp * op.k_noise, tf.amp * op.k_noise)
            if min_flag == 1:
                if p.value < p.best_value:
                    p.best_value = p.value
                    p.best_coord = p.coordinates
                if func_best > p.best_value:
                    best_x = p.coordinates.copy()
                    func_best = p.best_value
            else:
                if p.value > p.best_value:
                    p.best_value = p.value
                    p.best_coord = p.coordinates
                if func_best < p.best_value:
                    best_x = p.coordinates.copy()
                    func_best = p.best_value
    return best_x, func_best, iteration, best_chart, mean_chart


def initialization(op, tf):
    if isinstance(tf.down, (int, float)) and isinstance(tf.high, (int, float)):
        down = np.array([tf.down for _ in range(tf.dim)])
        high = np.array([tf.high for _ in range(tf.dim)])
    else:
        down = tf.down
        high = tf.high

    x = np.zeros((op.number_points,), dtype=object)
    for i in range(op.number_points):
        c = np.array([np.random.uniform(down[j], high[j]) for j in range(tf.dim)])
        val = tf.get_value(c) + np.random.uniform(-tf.amp * op.k_noise, tf.amp * op.k_noise)
        vel = np.array([np.random.uniform(down[j], high[j]) for j in range(tf.dim)])
        x[i] = Point(c=c, val=val, vel=vel, bv=val)
    return x


def get_best_point(points, min_flag):
    res = sorted(points, key=operator.attrgetter('_value'))
    if min_flag == 1:
        return res[0]
    else:
        return res[-1]


def update_func_value(points, tf, kn, is_first=False):
    for p in points():
        p.value = tf.get_value(p.coordinates) + np.random.uniform(-tf.amp * kn, tf.amp * kn)
        if is_first:
            p.best_value = p.value


def main():
    TEST_FUNC_2 = {
        "dimension": 2,
        "type": "bf",
        "number_extrema": 10,
        "coordinates": [
            [4, 2], [-3, -2], [-5, 3], [3, -3], [3, 5],
            [-2, 4], [0, -4], [5, -5], [-4, -4], [1, -1]
        ],
        "func_values": [0, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        "degree_smoothness": [
            [0.7, 0.9], [0.9, 0.6], [1.2, 0.3], [0.6, 1.3], [1.5, 2],
            [0.5, 0.9], [2, 0.6], [1.7, 1.1], [1.1, 0.8], [0.6, 1.1]
        ],
        "coefficients_abruptness": [
            [7, 7], [4, 5], [6, 6], [5, 7], [3.5, 5],
            [7, 3], [6, 5], [3, 6.3], [4.5, 5], [2, 3]
        ],
        "constraints_high": [6, 6],
        "constraints_down": [-6, -6],
        "global_min": [4, 2],
        "global_max": [2, -6],
        "amp_noise": 15.755,
        "min_value": 0.0,
        "max_value": 31.51
    }
    from problem.testfunc import TestFunction
    tf = TestFunction(**TEST_FUNC_2)

    d = {'n': [20, 30, 40], 'o': [0.2, 0.4, 0.6, 0.8]}
    op = PSAOptions(np=10, ni=100, kn=0, omega=1, fi_p=0.5, fi_g=1)
    alg = StandardPSA(op)
    p = alg.probability_estimate(tf, op, d, ep=0.2, number_runs=100, min_flag=1)
    print(p)


if __name__ == '__main__':
    main()
