# pylint: disable=invalid-name

import tensorflow as tf


class LearningRateMultiplier(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper for per layer learning rate.
    This wrapper is used to add per layer learning rates by
    providing per layer factors which are multiplied with the
    learning rate of the optimizer.

    Note: This is a wrapper and does not implement any
    optimization algorithm.

    For example, to make the multiplier for all the convolutional layers
    in a model 0.5, you can use the following.

    .. code-block:: python

        lr_multipliers = {}
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                for weights in layer.trainable_weights:
                    lr_multipliers[weights.name] = 0.5
        model.compile(
            optimizer=LearningRateMultiplier(
                optimizer=tf.keras.optimizers.SGD(1e-3),
                lr_multipliers=lr_multipliers
            )
        )

    Args:
        optimizer: An optimizer class to be wrapped.
        lr_multipliers: Dictionary of the per layer factors. For
            example `optimizer={'conv_1': 0.5, 'conv_1': 0.1}`.
        **kwargs: The arguments for instantiating the wrapped optimizer
            class.
    """
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, lr_multipliers: dict, **kwargs):
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            optimizer = tf.keras.utils.deserialize_keras_object(optimizer,
                                                                module_objects=tf.keras.optimizers,
                                                                custom_objects=kwargs.get(
                                                                    'custom_objects', None))
        self._optimizer = optimizer
        self._lr_multipliers = lr_multipliers or {}
        if 'name' not in kwargs:
            kwargs['name'] = f'LearningRateMultiplier_{optimizer._name}'
        for attr in ['_create_slots', '_stateless_fn', '_stateful_fn']:
            if hasattr(self._optimizer, attr):
                setattr(self, attr, getattr(self._optimizer, attr))
        super().__init__(**kwargs)

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, lr):
        self.optimizer.lr = lr

    def get_config(self):
        return {
            'optimizer': tf.keras.utils.serialize_keras_object(self._optimizer),
            'lr_multipliers': self._lr_multipliers
        }

    def _resource_apply_dense(self, grad, var, apply_state=None):
        multiplier = self._lr_multipliers.get(var.name, 1)
        if multiplier != 1:
            grad = tf.multiply(grad, tf.constant(multiplier, dtype=grad.dtype))
        # pylint: disable=protected-access
        return self._optimizer._resource_apply_dense(grad=grad, var=var, apply_state=apply_state)

    def _resource_apply_sparse(self, grad, var, apply_state=None):
        multiplier = self._lr_multipliers.get(var.name, 1)
        if multiplier != 1:
            grad = tf.multiply(grad, tf.constant(multiplier, dtype=grad.dtype))
        # pylint: disable=protected-access
        return self._optimizer._resource_apply_sparse(grad=grad, var=var, apply_state=apply_state)


tf.keras.utils.get_custom_objects().update({'LearningRateMultiplier': LearningRateMultiplier})
