class Object:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return Object()

    def __dir__(self):
        return []

    def __call__(self, *args, **kwargs):
        return Object()

    def __mro_entries__(self, _):
        return (Object(),)


backend = Object()
layers = Object()
models = Object()
utils = Object()
optimizers = Object()
