import numpy as np

import support
from algorithm.algabc import GSA, Options
from problem.testfunc import TestFunction


# TODO: добавить воздможность выбора метода останова (по умолчанию - итерации) среднеквадратичное откл от лучшей точки
class GSAOptions(Options):
    _alias_map = {
        'g_idx': ['gi', 'g_index'],  # -
        'g_zero': ['gz'],  # +
        'alpha': ['a'],  # +
        'gamma': ['g'],
        'elite_probe': ['ep'],  # -
        'r_norm': ['rn'],  # -
        'r_power': ['rp'],  # -
        'delta': ['d'],  # -
    }
    _required_keys = ('g_zero', 'alpha')

    def __init__(self, **kwargs):  # n, ni, ig, g0, alpha, gamma, ep=True, rn=2, rp=1, kn=0, delta=pow(10, -4)
        kw = support.normalize_kwargs(kwargs,
                                      alias_map=GSAOptions._alias_map,
                                      required=GSAOptions._required_keys)
        super().__init__(**kw)
        self._g_idx = 1 if 'g_idx' not in kw else kw['g_idx']
        self._g0 = kw['g_zero']
        self._alpha = kw['alpha']
        self._elite_probe = True if 'elite_probe' not in kw else kw['elite_probe']
        self._rn = 2 if 'r_norm' not in kw else kw['r_norm']
        self._rp = 1 if 'r_power' not in kw else kw['r_power']
        self._gamma = None if 'gamma' not in kw else kw['gamma']  # FIXME: зачем?
        self._delta = pow(10, -4) if 'delta' not in kw else kw['delta']  # останов

    def get_g_value(self, i, max_iter):
        if self._g_idx == 1:
            return self._g0 * np.exp(-self._alpha * i / max_iter)
        elif self._g_idx == 2:
            if self._gamma is not None:
                return self._g0 / (self._alpha + i**self._gamma)
            else:
                ValueError('Атрибут _gamma не установлен.')
        else:
            ValueError('Функции с таким индексом не существует: ' + str(self._g_idx))

    def update_op(self, **kwargs):
        kw = support.normalize_kwargs(kwargs, alias_map=GSAOptions._alias_map)
        for k, v in kw.items():
            print(k, v)
            if k in GSAOptions._alias_map:
                self.__setattr__(k, v)
        super().update_op(**kw)

    def __repr__(self):
        return (f'GSAOptions(n={self._number_points}, ni={self._number_iter}, '
                f'gi={self._g_idx}, gz={self._g0}, a={self._alpha}, g={self._gamma}, '
                f'ep={self._elite_probe}, rn={self._rn}, rp={self._rp}, d={self._delta}, kn={self._k_noise})')

    @property
    def g0(self):
        return self._g0

    @g0.setter
    def g0(self, v):
        self._g0 = v

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, v):
        self._alpha = v

    @property
    def g_idx(self):
        return self._g_idx

    @g_idx.setter
    def g_idx(self, v):
        self._g_idx = v

    @property
    def elite_probe(self):
        return self._elite_probe

    @elite_probe.setter
    def elite_probe(self, v):
        self._elite_probe = v

    @property
    def rn(self):
        return self._rn

    @rn.setter
    def rn(self, v):
        self._rn = v

    @property
    def rp(self):
        return self._rp

    @rp.setter
    def rp(self, v):
        self._rp = v

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, v):
        self._gamma = v

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, v):
        self._delta = v


class StandardGSA(GSA):
    def __init__(self, op, **kwargs):
        super().__init__(op=op, **kwargs)
        self._name = 'Standard GSA'
        self._full_name = 'Standard gravity search algorithm'

    def optimization(self, tf, min_flag=1):
        if self._options:
            return gsa(self._options, tf, min_flag)
        raise ValueError('Не установлены параметры алгоритма')


def initialization(n, dim, down, high):
    if isinstance(down, (int, float)) and isinstance(high, (int, float)):
        return np.random.uniform(down, high, (n, dim))
    elif isinstance(down, (list, tuple)) and isinstance(high, (list, tuple)):
        x = np.random.uniform(0, 1, (n, dim))
        for i in range(dim):
            x[:, i] = x[:, i] * (high[i] - down[i]) + down[i]
        return x
    else:
        raise ValueError('Некорректные down или high: down=' + str(down) + '; high=' + str(high))


def get_eval_func_val(x, tf, kn):
    f = np.array([tf.get_value(xi) + np.random.uniform(-tf.amp * kn, tf.amp * kn) for xi in x])
    return f


def space_bound(x, down, high):
    dim = len(x[0])
    for i in range(len(x)):
        high_border = x[i, :] > high
        down_border = x[i, :] < down
        # ~ - операция конвертации значения в противоположное
        x[i, :] = (x[i, :] * (~(down_border + high_border))) + (
            np.random.uniform(down, high, (1, dim)) * (down_border + high_border))
    return x


def find_mass(fit, min_flag):
    fit_max = np.max(fit)
    fit_min = np.min(fit)

    if fit_max == fit_min:
        mass = np.ones((len(fit), ))
    else:
        if min_flag == 1:  # minimization
            best = fit_min
            worst = fit_max
        else:  # maximization
            best = fit_max
            worst = fit_min

        mass = (fit - best) / (worst - best)

    mass = mass / np.sum(mass)

    return mass


def find_acceleration(x, mass, g, r_norm, r_power, ec, iter, max_iter):
    dim = len(x[0])
    n = len(x)

    final_per = 2
    if ec:
        k_best = final_per + (1 - iter / max_iter) * (100 - final_per)
        k_best = round(n * k_best / 100)
    else:
        k_best = n

    ds = np.argsort(mass)[::-1]
    E = np.zeros((n, dim))

    for i in range(n):
        for j in range(k_best):
            k = ds[j]
            if k != i:
                radius = np.linalg.norm(x[i, :] - x[k, :], r_norm)
                E[i] += np.random.uniform(0, 1) * mass[k] * ((x[k] - x[i]) / (
                    np.power(radius, r_power) + np.finfo(float).eps))

    return E * g


def move(x, a, last_v):
    v = np.random.uniform(0, 1, (len(x), len(x[0]))) * last_v + a
    new_x = x + v
    return new_x, v


def gsa(op, tf, min_flag):
    x = initialization(op.number_points, tf.dim, tf.down, tf.high)
    velocity = np.zeros((op.number_points, tf.dim))

    best_chart = []
    mean_chart = []
    func_best = None
    agent_best = None
    iteration = 0
    for i in range(op.number_iter):
        iteration = i + 1
        x = space_bound(x, tf.down, tf.high)
        fit = get_eval_func_val(x, tf, op.k_noise)

        if min_flag == 1:
            best = np.min(fit)
            best_x = np.argmin(fit)
        else:
            best = np.max(fit)
            best_x = np.argmax(fit)

        if i == 0:
            func_best = best
            agent_best = x[best_x, :]

        if min_flag == 1:
            if best < func_best:
                func_best = best
                agent_best = x[best_x, :]
        else:
            if best > func_best:
                func_best = best
                agent_best = x[best_x, :]

        best_chart.append(func_best)
        mean_chart.append(np.mean(fit))

        if op.delta is not None:
            ar_std = np.std(x, axis=0, ddof=1)
            _std = np.power(np.sum(np.power(ar_std, 2)), 0.5)
            if _std <= op.delta:
                return agent_best, func_best, iteration, best_chart, mean_chart

        mass = find_mass(fit, min_flag)
        g = op.get_g_value(iteration, op.number_iter)
        a = find_acceleration(x, mass, g, op.rn, op.rp, op.elite_probe, i, op.number_iter)
        x, velocity = move(x, a, velocity)
    return agent_best, func_best, iteration, best_chart, mean_chart


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
    tf = TestFunction(**TEST_FUNC_2)

    d = {'n': [50, 60, 70, 80]}
    op = GSAOptions(np=10, ni=50, g_idx=2, g_zero=100, alpha=20, gamma=2)
    alg = StandardGSA(op)
    p = alg.probability_estimate(tf, op, d, ep=0.2, number_runs=100, min_flag=1)
    print(p)


if __name__ == '__main__':
    main()

