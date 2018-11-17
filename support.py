import warnings


def normalize_kwargs(kw, alias_map=None, required=()):
    # по убыванию приоритета
    res = dict()
    if alias_map is None:
        alias_map = dict()
    for canonical, alias in alias_map.items():
        values, seen_key = [], []
        if canonical not in alias:
            if canonical in kw:
                values.append(kw.pop(canonical))
                seen_key.append(canonical)
        for a in alias:
            if a in kw:
                values.append(kw.pop(a))
                seen_key.append(a)
        if values:
            res[canonical] = values[0]
            if len(values) > 1:
                warnings.warn(f'Все псевдонимы kwargs {seen_key!r} относятся к '
                              f'{canonical!r}. Будет применен только {seen_key[0]!r}')
    res.update(kw)
    fail_keys = [k for k in required if k not in res]
    if fail_keys:
        raise TypeError(f"Требуются ключи {fail_keys} в kwargs")
    return res
