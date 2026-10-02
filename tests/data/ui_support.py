"""Stand-in for trame's State: item and attribute access plus has()."""


class FakeState(dict):
    __getattr__ = dict.__getitem__

    def has(self, key):
        return key in self
