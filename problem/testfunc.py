import numpy as np

from support import normalize_kwargs

_alias_map = {
    'index': ['idx'],
    'f_type': ['type', 'ft'],
    'coord': ['coordinates', 'c'],
    'func_val': ['func_values', 'fv'],
    'ds': ['degree_smoothness'],
    'sc': ['slope_coefficients', 'coefficients_abruptness'],
    'down': ['constraints_down', 'cd'],
    'high': ['constraints_high', 'ch'],
    'global_min': ['g_min'],
    'global_max': ['g_max'],
    'max_val': ['max_value'],
    'min_val': ['min_value'],
}

_required_keys = ('f_type', 'coord', 'func_val', 'ds', 'sc', 'down',
                  'high',)  # 'global_min', 'global_max', 'min_val', 'max_val'


# TODO: добавить создание случаной тестовой функции

class TestFunction:
    def __init__(self, **kwargs):
        kw = normalize_kwargs(kwargs, alias_map=_alias_map, required=('f_type', ))
        self._type = kw['f_type']
        if 'index' in kw:
            self._idx = kw['index']
        self._amp = None
        self._func = None

        if self._type in FUNCTIONS.keys():
            kw = normalize_kwargs(kw, alias_map=_alias_map, required=_required_keys)

            self._coord = kw['coord']
            self._func_val = kw['func_val']
            self._ds = kw['ds']
            self._sc = kw['sc']
            self._down = kw['down']
            self._high = kw['high']
            self._global_min = None if 'global_min' not in kw else kw['global_min']
            self._global_max = None if 'global_max' not in kw else kw['global_max']
            self._min_val = None if 'min_val' not in kw else kw['min_val']
            self._max_val = None if 'max_val' not in kw else kw['max_val']

            self._dim = len(self._coord[0])
            self._num_extrema = len(self._coord)

            if 'amp' in kw:
                self._amp = kw['amp']
            elif (self._min_val is not None) and (self._max_val is not None):
                self._amp = (self._min_val + self._max_val) / 2

            self.generate_func()
        elif self._type in ('arb', 'arbitrary'):
            pass

    def generate_func(self):
        # TODO: может отдельно создать функцию проверки данных в этом классе
        if not self._type:
            raise ValueError('Не задан тип функции')
        elif (self._coord is None) or (self._func_val is None):
            raise ValueError('Не заданы координаты или значения экстремумов')
        elif (self._ds is None) or (self._sc is None):
            raise ValueError('Не заданы степень гладкости или коэффициенты крутости')

        if self._type in FUNCTIONS.keys():
            self._func = FUNCTIONS[self._type](a=self._sc, c=self._coord,
                                               p=self._ds, b=self._func_val)
        else:
            raise ValueError('Неизвестный тип функции')

    def get_value(self, x):
        return self._func(x)

    def in_vicinity(self, x, epsilon, flag='min', strict_inequality=False):
        if flag == 'max':
            g_extrema = self._global_max
        else:
            g_extrema = self._global_min

        if strict_inequality:
            res = (np.abs(g_extrema - x) < epsilon).all()
        else:
            res = (np.abs(g_extrema - x) <= epsilon).all()
        return res

    # @classmethod
    # def from_json(cls, file_name):
    #     with open(file_name) as f:
    #         data = json.load(f)
    #     return cls.from_dict(data)

    # @classmethod
    # def from_dict(cls, d):
    #     for k, v in _field_names.items():
    #         if v not in d:
    #             raise KeyError(f'Нет обязательного поля: {v}')
    #     if not validate_type(d[_field_names['f_type']]):
    #         raise ValueError(f"Некорректное значение типа: {d['f_type']}. Доступные типы: {list(FUNCTIONS.keys())}")
    #     return cls(**d)

    def __repr__(self):
        # TODO: доделать
        pass

    def __str__(self):
        # TODO: доделать
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
        return self._ds

    @degree_smoothness.setter
    def degree_smoothness(self, val):
        self._ds = val

    @property
    def slope_coefficients(self):
        return self._sc

    @slope_coefficients.setter
    def slope_coefficients(self, val):
        self._sc = val

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
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, val):
        self._amp = val

    @property
    def global_min(self):
        return self._global_min

    @global_min.setter
    def global_min(self, val):
        self._global_min = val

    @property
    def global_max(self):
        return self._global_max

    @global_max.setter
    def global_max(self, val):
        self._global_max = val

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

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, f):
        self._func = f


def create_arbitrary_tf():
    # создание произвольной функции
    pass


# down, high - границы области (генерация координат)
# dim - размерность задачи,
# number_extrema - количество экстремумов,
# min_flag - минимумы или максимумы
# f_type - тип функции,
# ds_range - диапазон степеней гладкости,
# sc_range - диапазон коэффициентов крутости
# fv_range - диапазон значений функции в т экстремумах,
# global_extrema - (min, rand) - флаг
# ge_distance - различие глобального экст от остальных
def create_random_tf(ds_range, sc_range, fv_range, down, high, f_type='bf', dim=2, numb_ex=10, min_dist=0.5, ge_dist=3):
    if f_type not in FUNCTIONS.keys():
        raise ValueError(f'Некорректный тип функции {f_type}. Возможные типы: {list(FUNCTIONS.keys())}')

    # генерация координат
    if isinstance(down, (int, float)):
        down = np.array([down for _ in range(dim)])
    if isinstance(high, (int, float)):
        high = np.array([high for _ in range(dim)])
    c = np.zeros((numb_ex, dim))
    for i in range(numb_ex):
        is_duplicate = True
        while is_duplicate:
            c[i] = np.array([np.random.randint(down[j], high[j]) for j in range(dim)])
            is_duplicate = (c[i] == c[:i]).all(axis=1).any()
    c = np.array([np.array([np.random.randint(down[i], high[i]) for i in range(dim)]) for _ in range(numb_ex)])
    # генерация значений функции
    fv_range = np.float64(fv_range)
    f = np.around(sequence(fv_range, numb_ex, ge_d=ge_dist, distance=min_dist, ge_val='min'), decimals=2)  # для 'ep' fv_range = [15, 0]
    # генерация степеней гладкости
    p = np.around(np.random.uniform(ds_range[0], ds_range[1], (numb_ex, dim)), decimals=2)
    # генерация коэффициентов крутости
    a = np.around(np.random.uniform(sc_range[0], sc_range[1], (numb_ex, dim)), decimals=2)

    return TestFunction(f_type=f_type, coord=c, func_val=f, ds=p, sc=a,
                        down=down, high=high, global_min=c[0], min_val=f[0])


def sequence(s_range, n, ge_d=1, distance=0.5, ge_val='min'):
    ge_possible_values = ('min', 'rand')
    if ge_val not in ge_possible_values:
        raise ValueError(f'Некорректное значение аргумента g_extrema. '
                         f'Возможные значения: {ge_possible_values}')

    f = np.zeros(n)
    cut_len = abs(s_range[1] - s_range[0])
    min_distance = ge_d + (n - 1) * distance
    if not (cut_len > min_distance):
        raise ValueError(f'Диапазон варьирования слишком мал. '
                         f'Его длина должна быть > {min_distance}')
    reverse = True if s_range[0] > s_range[1] else False

    if ge_val in ge_possible_values[0]:
        f[0] = s_range[0]
    else:
        q = (cut_len - ge_d - (n - 1) * distance) / n
        high = s_range[0] - q if reverse else s_range[0] + q
        f[0] = np.random.uniform(s_range[0], high)

    for i in range(1, n):
        if i == 1:
            low = f[i - 1] - ge_d if reverse else f[i - 1] + ge_d
        else:
            low = f[i - 1] - distance if reverse else f[i - 1] + distance
        q = (cut_len - low - (n - i) * distance) / n
        high = low - q if reverse else low + q
        f[i] = np.random.uniform(low, high)
    return f


def method_min(a, c, p, b, **kwargs):
    # a - коэффициенты крутости
    # c - кординаты
    # p - степени гладкости
    # b - значения функции
    def func(x):
        l = np.zeros((len(b),))
        for i in range(len(b)):
            l[i] = np.sum(a[i] * np.abs(x - c[i]) ** p[i]) + b[i]
        return np.min(l)
    return func


def hyperbolic_potential_abs(a, c, p, b, **kwargs):
    def func(x):
        res = 0
        for i in range(len(b)):
            res += -1 / (np.sum(a[i] * np.abs(x - c[i]) ** p[i]) + b[i])
        return res
    return func


def hyperbolic_potential_sqr(a, c, b, **kwargs):
    def func(x):
        res = 0
        for i in range(len(b)):
            res += -1 / (np.sum(a[i] * (x - c[i]) ** 2) + b[i])
        return res
    return func


def exponential_potential(a, c, p, b, **kwargs):
    def func(x):
        res = 0
        for i in range(len(b)):
            res += (-b[i]) * np.exp((-a[i]) * np.sum(np.abs(x - c[i]) ** p[i]))
        return res
    return func


FUNCTIONS = {
    'bf': method_min,
    'hp_abs': hyperbolic_potential_abs,
    'hp_sqr': hyperbolic_potential_sqr,
    'ep': exponential_potential
}


def validate_type(t: str) -> bool:
    if t in FUNCTIONS.keys():
        return True
    return False


def validate_func_inform(t, a, c, p, b, **kwargs):
    # kwargs['idx']
    # kwargs['down']
    # kwargs['high']
    # kwargs['amp']
    # kwargs['real_min']
    # kwargs['real_max']
    # kwargs['min_val']
    # kwargs['max_val']
    if not isinstance(t, str) or t not in FUNCTIONS.keys():
        return False
    dim = len(c[0])
    n = len(c)
    if n != len(a) or n != len(p) or n != len(b):
        return False
    for i in range(n):
        if len(c[i]) != dim or len(p[i]) != dim:
            return False
    if t in ['hp_abs', 'ep']:
        pass
    elif t in ['bf', 'hp_sqr']:
        for item in a:
            if len(item) != dim:
                return False


def create_test_func(*args, **kwargs):
    for arg in args:
        if isinstance(arg, str) and arg in FUNCTIONS.keys():
            pass


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
    # print(f(x))
    # print(f1(x))
    # print(f2(x))
    # print(f3(x))

    TEST_FUNC_2 = {
        "dimension": 2,
        "type": 'bf',
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

    # tf = TestFunction(**TEST_FUNC_2)
    # print(tf.get_value(np.array([4, 2])))
    # print(tf.get_value(np.array([2, -6])))
    # print(tf.in_vicinity(np.array([4, 2.1]), epsilon=0.2))
    # print(tf.in_vicinity(np.array([3.79, 2.1]), epsilon=0.2))

    tf = create_random_tf([0.5, 2.2], [2.0, 8.0], [0, 15], -6, 6,
                          f_type='bf', dim=2, numb_ex=10, min_dist=0.5, ge_dist=3)
    print(tf.coord)
    print(tf.get_value(tf.coord[0]))


if __name__ == '__main__':
    main()
