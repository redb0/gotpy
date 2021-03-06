from typing import Union

import numpy as np

import support
from algorithm.algabc import Options, ASA
from problem.testfunc import TestFunction


class ASAOptions(Options):
    _alias_map = {
        'max_temp': ['max_t'],
        'min_temp': ['min_t'],
        'init_mode': ['im'],
        'idx_temp_func': ['itf'],
    }
    _required_keys = ('max_temp', 'min_temp')

    def __init__(self, **kwargs):
        kw = support.normalize_kwargs(kwargs,
                                      alias_map=ASAOptions._alias_map,
                                      required=ASAOptions._required_keys)
        print(kw)
        super().__init__(**kw)
        self._max_temp = kw['max_temp']
        self._min_temp = kw['min_temp']
        self._idx_temp_func = 1 if 'idx_temp_func' not in kw else kw['idx_temp_func']
        self._init_mode = 'rand' if 'init_mode' not in kw else kw['init_mode']

    def get_temperature(self, i: int) -> Union[int, float]:
        t = None
        if self._idx_temp_func == 1:
            t = boltzmann_annealing(i, self._max_temp)
        elif self._idx_temp_func == 2:
            t = linear_temp(i, self._max_temp)
        return t

    def update_op(self, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=ASAOptions._alias_map)
        for k, v in kw.items():
            print(k, v)
            if k in ASAOptions._alias_map:
                self.__setattr__(k, v)
        super().update_op(**kw)

    @property
    def max_temp(self) -> Union[int, float]:
        return self._max_temp

    @max_temp.setter
    def max_temp(self, v: Union[int, float]) -> None:
        self._max_temp = v

    @property
    def min_temp(self) -> Union[int, float]:
        return self._min_temp

    @min_temp.setter
    def min_temp(self, v: Union[int, float]) -> None:
        self._min_temp = v

    @property
    def init_mode(self) -> str:
        return self._init_mode

    @init_mode.setter
    def init_mode(self, v: str) -> None:
        self._init_mode = v

    @property
    def idx_temp_func(self) -> int:
        return self._idx_temp_func

    @idx_temp_func.setter
    def idx_temp_func(self, v: int) -> None:
        self._idx_temp_func = v

    def __repr__(self) -> str:
        return (f'ASAOptions(max_t={self._max_temp}, min_t={self._min_temp}, '
                f'itf={self._idx_temp_func}, im={self._init_mode})')

    def __str__(self):
        return (f'ASAOptions(max_temp={self._max_temp}, min_temp={self._min_temp}, '
                f'idx_temp_func={self._idx_temp_func}, init_mode={self._init_mode})')


class StandardASA(ASA):
    def __init__(self, op, **kwargs):
        super().__init__(op=op, **kwargs)
        self._name = 'Standard ASA'
        self._full_name = ''

    def optimization(self, tf, min_flag=1):  # asa
        if self._options:
            return asa(self._options, tf, min_flag)
        raise ValueError('Не установлены параметры алгоритма')


def asa(op, tf, min_flag):
    t = op.max_temp
    x = init_start_point(tf, op.init_mode)
    energy = get_eval_func_val(x, tf, op.k_noise)

    charts = []
    f = []

    i = 1
    while t > op.min_temp:
        test_x = init_test_point(x, t, tf)
        new_energy = get_eval_func_val(test_x, tf, op.k_noise)
        x, energy = transition(x, test_x, t, energy, new_energy)
        charts.append(x)
        f.append(energy)
        t = op.get_temperature(i)
        i += 1
    return x, energy, i, charts, f


def transition(x, test_x, temp, sys_energy, test_energy):
    delta_energy = test_energy - sys_energy
    new_x, new_energy = x, sys_energy
    if delta_energy < 0:
        new_x = test_x
        new_energy = test_energy
    else:
        transition_probability = 1 / (1 + np.exp(delta_energy / temp))
        if np.random.uniform(0, 1) < transition_probability:
            new_x = test_x
            new_energy = test_energy
    return new_x, new_energy


def get_eval_func_val(x, tf, kn):
    f = np.array([tf.get_value(x) + np.random.uniform(-tf.amp * kn, tf.amp * kn)])
    return f


def init_start_point(tf, init_mode):
    if init_mode == 'rand':
        return np.array([np.random.uniform(tf.down[i], tf.high[i]) for i in range(tf.dim)])


def init_test_point(x, temp, tf):
    check = True
    p = None
    while check:
        p = np.random.normal(x, temp, (tf.dim,))
        check = space_bound(p, tf.down, tf.high)
    return p


def space_bound(x, down, high):
    return (x > high).any() or (x < down).any()


def boltzmann_annealing(i, max_temp):
    return max_temp / np.log(1 + 2 * i)


def linear_temp(i, max_temp):
    return max_temp / i


def main():
    TEST_FUNC_2 = {
        "dimension": 2,
        "type": "bocharov_feldbaum",
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
    c = np.array([[4, 2], [-3, -2], [-5, 3], [3, -3], [3, 5],
                  [-2, 4], [0, -4], [5, -5], [-4, -4], [1, -1]])
    b = np.array([0, 3, 5, 6, 7, 8, 9, 10, 11, 12])
    a = np.array([[7, 7], [4, 5], [6, 6], [5, 7], [3.5, 5],
                  [7, 3], [6, 5], [3, 6.3], [4.5, 5], [2, 3]])
    p = np.array([[0.7, 0.9], [0.9, 0.6], [1.2, 0.3], [0.6, 1.3], [1.5, 2],
                  [0.5, 0.9], [2, 0.6], [1.7, 1.1], [1.1, 0.8], [0.6, 1.1]])
    g_min = np.array([4, 2])
    tf = TestFunction(f_type='bf', coord=c, func_val=b, ds=p, sc=a, down=[-6, -6], high=[6, 6],
                      global_min=g_min, global_max=np.array([2, -6]), min_val=0, max_val=31.51)
    tf.generate_func()

    d = {'max_temp': [350, 400, 450,  500, 550, 600]}
    op = ASAOptions(n=0, number_iter=100, max_temp=350, min_temp=0.1, idx_temp_func=2, init_mode='rand', kn=1)
    alg = StandardASA(op)
    p = alg.probability_estimate(tf, op, d, ep=0.2, number_runs=100, min_flag=1)
    print(p)


if __name__ == '__main__':
    main()
