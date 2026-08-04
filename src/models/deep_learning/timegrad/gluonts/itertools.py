def maybe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return None


class Cyclic:
    pass


class PseudoShuffled:
    pass


class Cached:
    pass
