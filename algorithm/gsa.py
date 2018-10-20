import numpy as np


from algorithm.algabc import GSA, Options
from problem.testfunc import TestFunction


class GSAOptions(Options):
    def __init__(self, n, ni, ig, g0, alpha, gamma, ep=True, rn=2, rp=1, kn=0):
        super().__init__(n, ni, kn)
        self._g_idx = ig
        self._g0 = g0
        self._alpha = alpha
        self._elite_probe = ep
        self._rn = rn
        self._rp = rp
        self._gamma = gamma  # FIXME: зачем?

    def get_g_value(self, i, max_iter):
        if self._g_idx == 1:
            return self._g0 * np.exp(-self._alpha * i / max_iter)
        else:
            ValueError('Функции с таким индексом не существует: ' + str(self._g_idx))

    @property
    def g0(self):
        return self._g0

    @property
    def alpha(self):
        return self._alpha

    @property
    def g_idx(self):
        return self._g_idx

    @property
    def elite_probe(self):
        return self._elite_probe

    @property
    def rn(self):
        return self._rn

    @property
    def rp(self):
        return self._rp

    @property
    def gamma(self):
        return self._gamma


class StandardGSA(GSA):
    def __init__(self, op, **kwargs):
        super().__init__(op=op, **kwargs)
        self._name = 'Standard GSA'
        self._full_name = 'Standard gravity search algorithm'

    def gsa(self, tf, min_flag=1):
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
    f = np.array([tf.get_value(xi) + np.random.uniform(-tf.amp_noise * kn, tf.amp_noise * kn) for xi in x])
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

    for i in range(op.ni):
        iteration = i + 1
        x = space_bound(x, tf.down, tf.high)
        fit = get_eval_func_val(x, tf, op.kn)

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

        mass = find_mass(fit, min_flag)
        g = op.get_g_value(iteration, op.ni)
        a = find_acceleration(x, mass, g, op.rn, op.rp, op.elite_probe, i, op.ni)
        x, velocity = move(x, a, velocity)
    return func_best, agent_best, best_chart, mean_chart


def main():
    n = 100
    max_iter = 100
    idx_g = 1
    g0 = 100
    alpha = 20
    gamma = None  # ???
    op = GSAOptions(n, max_iter, idx_g, g0, alpha, gamma, ep=True, rn=2, rp=1, kn=0)

    a = np.array([
        [7, 7], [4, 5], [6, 6], [5, 7], [3.5, 5],
        [7, 3], [6, 5], [3, 6.3], [4.5, 5], [2, 3]
    ])
    b = np.array([0, 5, 6, 6.5, 7, 8, 8.5, 9, 10, 11])
    c = np.array([
        [2, -3], [-4, 3], [4, 5], [-2, 1], [-3, -4],
        [-5, -3], [4, -2], [2, 2], [3, 5], [-1, -1]
    ])
    p = np.array([
        [0.7, 0.9], [0.9, 0.6], [1.2, 0.3], [0.6, 1.3], [1.5, 2],
        [0.5, 0.9], [2, 0.6], [1.7, 1.1], [1.1, 0.8], [0.6, 1.1]
    ])
    down = [-6, -6]
    high = [6, 6]

    tf = TestFunction(0, t='bf', c=c, f=b, p=p, a=a, down=down, high=high,
                      real_min=np.array([2, -3]), real_max=np.array([6, -6]), min_val=0, max_val=32.1)
    tf.generate_func()
    alg = StandardGSA(op)
    func_best, agent_best, best_chart, mean_chart = alg.gsa(tf, min_flag=1)
    print(func_best)
    print(agent_best)

if __name__ == '__main__':
    main()

