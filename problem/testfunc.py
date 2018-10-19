import numpy as np
from typing import List, Union

# Numb = Union[int, float]
# ListNumb = List[Numb]
# Matrix = List[ListNumb]


class TestFunction:
    def __init__(self, idx=0, t='',
                 c=None, f=None, p=None, a=None, down=None, high=None, an=None,
                 real_min=None, real_max=None, min_val=0, max_val=0):
        self._func = None
        self._idx = idx
        self._dim = len(c[0])
        self._num_extrema = len(c)
        self._type = t
        self._coord = c
        self._func_val = f
        self._degree_smoothness = p
        self._coef_abruptness = a
        self._down = down
        self._high = high
        self._amp_noise = an
        self._real_min = real_min
        self._real_max = real_max
        self._min_val = min_val
        self._max_val = max_val

    def generate_func(self):
        # TODO: может отдельно создать функцию проверки данных в этом классе
        if not self._type:
            raise ValueError('Не задан тип функции')
        elif (not self._coord) or (not self._func_val):
            raise ValueError('Не заданы координаты или значения экстремумов')
        elif (not self._degree_smoothness) or (not self._coef_abruptness):
            raise ValueError('Не заданы степень гладкости или коэффициенты крутости')

        if self._type == 'bf':  # feldbaum_function
            self._func = method_min(self._coef_abruptness, self._coord, self._degree_smoothness, self._func_val)
        elif self._type == 'hp_abs':  # hyperbolic_potential_abs
            self._func = hyperbolic_potential_abs(self._coef_abruptness, self._coord,
                                                  self._degree_smoothness, self._func_val)
        elif self._type == 'hp_sqr':  # hyperbolic_potential_sqr
            self._func = hyperbolic_potential_sqr(self._coef_abruptness, self._coord, self._func_val)
        elif self._type == 'ep':  # exponential_potential
            self._func = exponential_potential(self._coef_abruptness, self._coord,
                                               self._degree_smoothness, self._func_val)
        else:
            raise ValueError('Неизвестный тип функции')

    def get_value(self, x):
        pass

    def from_json(self, file_name):
        pass

    def from_dict(self, d):
        pass

    @property
    def func(self):
        return self._func

    @property
    def index(self):
        return self._idx

    @index.setter
    def index(self, val):
        if isinstance(val, int) and val > 0:
            self._idx = val
        else:
            raise ValueError('Индекс должен быть натуральным числом')

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, val):
        self._dim = val

    @property
    def number_extrema(self):
        return self._num_extrema

    @property
    def func_type(self):
        return self._type

    @func_type.setter
    def func_type(self, val):
        self._type = val

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, val):
        self._coord = val

    @property
    def func_val(self):
        return self._func_val

    @func_val.setter
    def func_val(self, val):
        self._func_val = val

    @property
    def degree_smoothness(self):
        return self._degree_smoothness

    @degree_smoothness.setter
    def degree_smoothness(self, val):
        self._degree_smoothness = val

    @property
    def coef_abruptness(self):
        return self._coef_abruptness

    @coef_abruptness.setter
    def coef_abruptness(self, val):
        self._coef_abruptness = val

    @property
    def down(self):
        return self._down

    @down.setter
    def down(self, val):
        self._down = val

    @property
    def high(self):
        return self._high

    @high.setter
    def high(self, val):
        self._high = val

    @property
    def amp_noise(self):
        return self._amp_noise

    @amp_noise.setter
    def amp_noise(self, val):
        self._amp_noise = val

    @property
    def real_min(self):
        return self._real_min

    @real_min.setter
    def real_min(self, val):
        self._real_min = val

    @property
    def real_max(self):
        return self._real_max

    @real_max.setter
    def real_max(self, val):
        self._real_max = val

    @property
    def min_val(self):
        return self._min_val

    @min_val.setter
    def min_val(self, val):
        self._min_val = val

    @property
    def max_val(self):
        return self._max_val

    @max_val.setter
    def max_val(self, val):
        self._max_val = val


def method_min(a, c, p, b):
    def func(x):
        l = np.zeros((len(b),))
        for i in range(len(b)):
            l[i] = np.sum(a[i] * np.abs(x - c[i]) ** p[i]) + b[i]
        return np.min(l)
    return func


def hyperbolic_potential_abs(a, c, p, b):
    def func(x):
        res = 0
        for i in range(len(b)):
            res += -1 / (np.sum(a[i] * np.abs(x - c[i]) ** p[i]) + b[i])
        return res
    return func


def hyperbolic_potential_sqr(a, c, b):
    def func(x):
        res = 0
        for i in range(len(b)):
            res += -1 / (np.sum(a[i] * (x - c[i]) ** 2) + b[i])
        return res
    return func


def exponential_potential(a, c, p, b):
    def func(x):
        res = 0
        for i in range(len(b)):
            res += (-b[i]) * np.exp((-a[i]) * np.sum(np.abs(x - c[i]) ** p[i]))
        return res
    return func


def main():
    a = np.array([[5,5],[4,5],[6,6],[5,6],[3.5,5],[7,3],[6,5],[3,6.3],[4.5,5],[2,3]])
    b = np.array([0,0.5,1,1.5,5,8,8.5,9,10,11])
    c = np.array([[2,-3],[-4,3],[4,5],[-2,1],[-3,-4],[-5,-3],[4,-2],[2,2],[3,5],[-1,-1]])
    p = np.array([[0.7,0.9],[0.9,0.6],[1.2,0.3],[0.6,1.3],[1.5,2],[0.5,0.9],[2,0.6],[1.7,1.1],[1.1,0.8],[0.6,1.1]])

    a1 = np.array([0.8, 2, 1, 1.5, 0.5])
    b1 = np.array([0.1, 0.2, 0.3, 0.4, 0.45])
    c1 = np.array([[0, 0], [2, 2], [4, 4], [-2, -2], [-4, -4]])
    p1 = np.array([[0.5, 0.6], [1, 0.8], [0.8, 1], [0.6, 0.9], [0.8, 0.8]])

    b2 = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85])

    x = np.array([0, 0])
    f = method_min(a, c, p, b)
    f1 = hyperbolic_potential_abs(a1, c1, p1, b1)
    f2 = hyperbolic_potential_sqr(a, c, b2)
    f3 = exponential_potential(a1, c1, p1, b1)
    print(f(x))
    print(f1(x))
    print(f2(x))
    print(f3(x))

if __name__ == '__main__':
    main()
