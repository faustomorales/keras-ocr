# pylint: disable=invalid-name,too-many-locals,too-many-arguments
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

from . import tools


def swish(x, beta=1):
    return x * keras.backend.sigmoid(beta * x)


keras.utils.get_custom_objects().update({'swish': keras.layers.Activation(swish)})


class SpatialTransformer(keras.layers.Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """
    def __init__(self, *args, **kwargs):
        self.localization_net = None
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        input_layer = keras.layers.Input(shape=input_shape[1:])
        x = keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(input_layer)
        x = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(6,
                               weights=[
                                   np.zeros((64, 6), dtype='float32'),
                                   np.float32([[1, 0, 0], [0, 1, 0]]).flatten()
                               ])(x)
        self.localization_net = keras.models.Model(inputs=input_layer, outputs=x)

    # pylint: disable=no-self-use
    def compute_output_shape(self, input_shape):
        return input_shape

    # pylint: disable=unused-argument,too-many-statements
    def call(self, X):
        locnet_x = X
        locnet_y = self.localization_net(X)
        output_size = locnet_x.shape[1:]
        batch_size = tf.shape(locnet_x)[0]
        height = tf.shape(locnet_x)[1]
        width = tf.shape(locnet_x)[2]
        num_channels = tf.shape(locnet_x)[3]

        locnet_y = tf.reshape(locnet_y, shape=(batch_size, 2, 3))

        locnet_y = tf.reshape(locnet_y, (-1, 2, 3))
        locnet_y = tf.cast(locnet_y, 'float32')

        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1])  # flatten?
        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, tf.stack([batch_size, 3, -1]))

        transformed_grid = tf.matmul(locnet_y, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x = tf.reshape(x_s, [-1])
        y = tf.reshape(y_s, [-1])

        # Interpolate
        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width = output_size[1]

        x = tf.cast(x, dtype='float32')
        y = tf.cast(y, dtype='float32')
        x = .5 * (x + 1.0) * width_float
        y = .5 * (y + 1.0) * height_float

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1, dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width * height
        pixels_batch = tf.range(batch_size) * flat_image_dimensions
        flat_output_dimensions = output_height * output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(locnet_x, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        transformed_image = tf.add_n([
            area_a * pixel_values_a, area_b * pixel_values_b, area_c * pixel_values_c,
            area_d * pixel_values_d
        ])
        # Finished interpolation

        transformed_image = tf.reshape(transformed_image,
                                       shape=(batch_size, output_height, output_width,
                                              num_channels))
        return transformed_image

    # pylint: disable=no-self-use
    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    # pylint: disable=no-self-use
    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, shape=(1, -1))
        y_coordinates = tf.reshape(y_coordinates, shape=(1, -1))
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid


def CTCDecoder(rnn_steps_to_discard=2):  # pylint: disable=invalid-name
    def decoder(y_pred):
        y_pred = y_pred[:, rnn_steps_to_discard:, :]
        input_length = keras.backend.ones_like(y_pred[:, 0, 0]) * keras.backend.cast(
            keras.backend.shape(y_pred)[1], 'float32')
        return keras.backend.ctc_decode(y_pred, input_length)[0]

    return keras.layers.Lambda(decoder, name='decode')


def make_ctc_loss(rnn_steps_to_discard=2):
    """Make a CTC loss function for the recognizer.

    Args:
        rnn_steps_to_discard: The number of initial steps
            to discard from the RNN.
    """
    def loss(inputs):
        y_pred, y_true = inputs
        y_pred = y_pred[:, rnn_steps_to_discard:, :]
        label_length = keras.backend.sum(keras.backend.cast(
            keras.backend.greater(y_true, keras.backend.constant(-1)), 'float32'),
                                         axis=1,
                                         keepdims=True)
        input_length = keras.backend.ones_like(y_pred[:, 0:1, 0],
                                               name='ctc_ones') * keras.backend.cast(
                                                   keras.backend.shape(y_pred)[1], 'float32')
        return keras.backend.ctc_batch_cost(y_true=y_true,
                                            y_pred=y_pred,
                                            input_length=input_length,
                                            label_length=label_length)

    return loss


def build_model(alphabet,
                height,
                width,
                color=False,
                filters=None,
                rnn_units=None,
                dropout=0.25,
                rnn_steps_to_discard=2,
                pool_size=2,
                stn=True):
    inputs = keras.layers.Input((height, width, 3 if color else 1))
    x = keras.layers.Permute((2, 1, 3))(inputs)
    x = keras.layers.Lambda(lambda x: x[:, :, ::-1])(x)
    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu', padding='same', name='conv_1')(x)
    x = keras.layers.Conv2D(filters[1], (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = keras.layers.Conv2D(filters[2], (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = keras.layers.BatchNormalization(name='bn_3')(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='maxpool_3')(x)
    x = keras.layers.Conv2D(filters[3], (3, 3), activation='relu', padding='same', name='conv_4')(x)
    x = keras.layers.Conv2D(filters[4], (3, 3), activation='relu', padding='same', name='conv_5')(x)
    x = keras.layers.BatchNormalization(name='bn_5')(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='maxpool_5')(x)
    x = keras.layers.Conv2D(filters[5], (3, 3), activation='relu', padding='same', name='conv_6')(x)
    x = keras.layers.Conv2D(filters[6], (3, 3), activation='relu', padding='same', name='conv_7')(x)
    x = keras.layers.BatchNormalization(name='bn_7')(x)
    if stn:
        x = SpatialTransformer()(x)
    x = keras.layers.Reshape(target_shape=(width // pool_size**2,
                                           (height // pool_size**2) * filters[-1]),
                             name='reshape')(x)

    x = keras.layers.Dense(rnn_units[0], activation='relu', name='fc_9')(x)

    rnn_1_forward = keras.layers.LSTM(rnn_units[0],
                                      kernel_initializer="he_normal",
                                      return_sequences=True,
                                      name='lstm_10')(x)
    rnn_1_back = keras.layers.LSTM(rnn_units[0],
                                   kernel_initializer="he_normal",
                                   go_backwards=True,
                                   return_sequences=True,
                                   name='lstm_10_back')(x)
    rnn_1_add = keras.layers.Add()([rnn_1_forward, rnn_1_back])
    rnn_2_forward = keras.layers.LSTM(rnn_units[1],
                                      kernel_initializer="he_normal",
                                      return_sequences=True,
                                      name='lstm_11')(rnn_1_add)
    rnn_2_back = keras.layers.LSTM(rnn_units[1],
                                   kernel_initializer="he_normal",
                                   go_backwards=True,
                                   return_sequences=True,
                                   name='lstm_11_back')(rnn_1_add)
    x = keras.layers.Concatenate()([rnn_2_forward, rnn_2_back])
    x = keras.layers.Dropout(dropout, name='dropout')(x)
    x = keras.layers.Dense(len(alphabet) + 1,
                           kernel_initializer='he_normal',
                           activation='softmax',
                           name='fc_12')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    prediction_model = keras.models.Model(
        inputs=inputs, outputs=CTCDecoder(rnn_steps_to_discard=rnn_steps_to_discard)(model.output))
    labels = keras.layers.Input(name='labels',
                                shape=[model.output_shape[1] - rnn_steps_to_discard],
                                dtype='float32')
    loss = keras.layers.Lambda(make_ctc_loss(rnn_steps_to_discard=rnn_steps_to_discard),
                               name='loss')([model.output, labels])
    training_model = keras.models.Model(inputs=[model.input, labels], outputs=loss)
    training_model.compile(loss=lambda _, y_pred: y_pred, optimizer=keras.optimizers.RMSprop())
    return model, training_model, prediction_model


class Recognizer:
    def __init__(self,
                 alphabet,
                 height,
                 width,
                 color=False,
                 filters=None,
                 rnn_units=None,
                 dropout=0.25,
                 rnn_steps_to_discard=2):
        if filters is None:
            filters = [64, 128, 256, 256, 512, 512, 512]
        if rnn_units is None:
            rnn_units = [128, 128]
        self.alphabet = alphabet
        self.model, self.training_model, self.prediction_model = build_model(
            alphabet=alphabet,
            height=height,
            width=width,
            rnn_steps_to_discard=rnn_steps_to_discard,
            color=color,
            filters=filters,
            rnn_units=rnn_units,
            dropout=dropout)

    def get_batch_generator(self, image_generator, batch_size=8):
        """
        Generate batches of training data.

        Args:
            image_generator: An image / sentence tuple generator. The images should
                be in color even if the OCR is setup to handle grayscale as they
                will be converted here.
            batch_size: How many images to generate at a time.

        ```python
            text_generator = recognizer.get_text_generator(max_string_length=8)
            image_generator = recognizer.get_image_generator(
                font_groups={
                    'characters': [
                        'Century Schoolbook',
                        'Courier', 'STIX',
                        'URW Chancery L',
                        'FreeMono'
                    ]
                },
                text_generator=text_generator,
                font_size=18
            )
            recognizer.get_training_generator(image_generator=image_generator)
        ```
        """
        y = np.zeros((batch_size, 1))
        blank_label = -1
        if self.training_model is None:
            raise Exception('You must first call create_training_model().')
        max_string_length = self.training_model.input_shape[1][1]

        while True:
            batch = [data for data, _ in zip(image_generator, range(batch_size))]
            if not self.model.input_shape[-1] == 3:
                images = [
                    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
                    for image, _, _ in batch
                ]
            else:
                images = [image for image, _, _ in batch]
            images = np.array([image.astype('float32') / 255 for image in images])
            sentences = [sentence for _, sentence, _ in batch]
            assert all(c in self.alphabet
                       for c in ''.join(sentences)), 'Found illegal characters in sentence.'
            assert all(sentences), 'Found a zero length sentence.'
            assert all(
                len(sentence) <= max_string_length
                for sentence in sentences), 'A sentence is longer than this model can predict.'
            labels = np.array([[self.alphabet.index(c) for c in sentence] + [blank_label] *
                               (max_string_length - len(sentence)) for sentence in sentences])
            yield [images, labels], y

    def recognize(self, image):
        image = tools.read_and_fit(filepath_or_array=image,
                                   width=self.prediction_model.input_shape[2],
                                   height=self.prediction_model.input_shape[1],
                                   cval=0)
        if self.prediction_model.input_shape[-1] == 1:
            image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        image = image.astype('float32') / 255
        return ''.join([
            self.alphabet[idx] for idx in self.prediction_model.predict(image[np.newaxis])[0]
            if idx != -1
        ])
