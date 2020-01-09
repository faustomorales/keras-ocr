import tensorflow as tf
import numpy as np

import keras_ocr


def test_per_layer_lr_multiplier():
    inputs = tf.keras.layers.Input((100, 100, 3))
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same',
                               name='frozen1')(inputs)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', name='free1')(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Dense(8, name='frozen2')(x)
    x = tf.keras.layers.Dense(8, name='free2')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    X = np.random.randn(1, 100, 100, 3)
    y_true = np.random.randn(1, 8)

    weights_before = {weights.name: weights.numpy() for weights in model.trainable_weights}
    lr_multipliers1 = {}
    lr_multipliers2 = {}
    for layer in model.layers:
        if 'frozen' in layer.name:
            for weights in layer.trainable_weights:
                # lr_multipliers1 sets the learning rates for frozen
                # layers to zero.
                lr_multipliers1[weights.name] = 0

                # lr_multipliers2 sets the learning rates for frozen
                # layers to be very small.
                lr_multipliers2[weights.name] = 1e-3

    # We want to make sure that serialization works so
    # we try it both with a directly instantiated optimizer
    # as well as a deserialized wrapper.
    optimizer1 = keras_ocr.custom_objects.LearningRateMultiplier(
        optimizer=tf.keras.optimizers.SGD(1e-3), lr_multipliers=lr_multipliers1)
    optimizer2 = tf.keras.utils.deserialize_keras_object(
        tf.keras.utils.serialize_keras_object(optimizer1))
    optimizer3 = keras_ocr.custom_objects.LearningRateMultiplier(
        optimizer=tf.keras.optimizers.SGD(1e-3), lr_multipliers=lr_multipliers2)
    optimizer4 = tf.keras.utils.deserialize_keras_object(
        tf.keras.utils.serialize_keras_object(optimizer3))
    for zero, optimizer in [(True, optimizer1), (True, optimizer2), (False, optimizer3),
                            (False, optimizer4)]:
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(X, y_true, epochs=10, verbose=0)
        weights_after = {weights.name: weights.numpy() for weights in model.trainable_weights}
        delta_frozen = []
        delta_free = []
        for k in weights_before:
            delta = np.abs(weights_after[k] - weights_before[k]).max()
            if k in lr_multipliers1:
                delta_frozen.append(delta)
            else:
                delta_free.append(delta)
        assert np.max(delta_free) > 0
        if zero:
            # There should have been no change in the frozen weights.
            assert np.max(delta_frozen) == 0
        else:
            # We should have some change in weights.
            assert np.max(delta_frozen) > 0

            # That change should be very, very small relative
            # to the change in the layers with the full learning
            # rate.
            assert np.max(delta_frozen) < 1e-2 * np.max(delta_free)
        model.set_weights(list(weights_before.values()))
