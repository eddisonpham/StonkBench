def validated(*args, **kwargs):
    def _decorator(c):
        return c
    if args and callable(args[0]) and not isinstance(args[0], type):
        return _decorator(args[0])
    return _decorator


def dummy_obj(obj):
    return obj
