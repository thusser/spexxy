class MockParameter(object):
    def __init__(self, val):
        self.name = 'parameter'
        self.value = val


class MockComponent(object):
    def __init__(self, name):
        self.name = name
        self.prefix = name
        self.param_hints = {}

    @property
    def param_names(self):
        return self.param_hints.keys()

    def set_param_hint(self, name, **opts):
        if name not in self.param_hints:
            self.param_hints[name] = {}
        self.param_hints[name].update(**opts)
