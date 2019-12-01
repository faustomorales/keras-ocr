# pylint: disable=invalid-name,too-many-locals,too-many-arguments
import typing
import string

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

from . import tools

PRETRAINED_WEIGHTS = {
    # Keys are (weights_name, include_top, filters, rnn_units, color, stn)
    # pylint: disable=line-too-long
    ('kurapan', True, (64, 128, 256, 256, 512, 512, 512), (128, 128), False, True): {
        'url': 'https://storage.googleapis.com/keras-ocr/crnn_kurapan.h5',
        'sha256': 'a7d8086ac8f5c3d6a0a828f7d6fbabcaf815415dd125c32533013f85603be46d',
        'alphabet': string.digits + string.ascii_lowercase
    },
    ('kurapan', False, (64, 128, 256, 256, 512, 512, 512), (128, 128), False, True): {
        'url': 'https://storage.googleapis.com/keras-ocr/crnn_kurapan_notop.h5',
        'sha256': '027fd2cced3cbea0c4f5894bb8e9e85bac04f11daf96b8fdcf1e4ee95dcf51b9',
        'alphabet': None
    }
}


def swish(x, beta=1):
    return x * keras.backend.sigmoid(beta * x)


keras.utils.get_custom_objects().update({'swish': keras.layers.Activation(swish)})


def _repeat(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='int32')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])


def _meshgrid(height, width):
    x_linspace = tf.linspace(-1., 1., width)
    y_linspace = tf.linspace(-1., 1., height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates = tf.reshape(x_coordinates, shape=(1, -1))
    y_coordinates = tf.reshape(y_coordinates, shape=(1, -1))
    ones = tf.ones_like(x_coordinates)
    indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
    return indices_grid


# pylint: disable=too-many-statements
def _transform(inputs):
    locnet_x, locnet_y = inputs
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
    indices_grid = _meshgrid(output_height, output_width)
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
    base = _repeat(pixels_batch, flat_output_dimensions)
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
                                   shape=(batch_size, output_height, output_width, num_channels))
    return transformed_image


def CTCDecoder():  # pylint: disable=invalid-name
    def decoder(y_pred):
        y_pred = y_pred[:, :, :]
        input_length = keras.backend.ones_like(y_pred[:, 0, 0]) * keras.backend.cast(
            keras.backend.shape(y_pred)[1], 'float32')
        return keras.backend.ctc_decode(y_pred, input_length)[0]

    return keras.layers.Lambda(decoder, name='decode')


def build_model(alphabet,
                height,
                width,
                color=False,
                filters=None,
                rnn_units=None,
                dropout=0.25,
                rnn_steps_to_discard=2,
                optimizer=None,
                pool_size=2,
                stn=True):
    if optimizer is None:
        optimizer = keras.optimizers.SGD(decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
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
        # pylint: disable=pointless-string-statement
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
        stn_input_output_shape = (width // pool_size**2, height // pool_size**2, filters[6])
        stn_input_layer = keras.layers.Input(shape=stn_input_output_shape)
        locnet_y = keras.layers.Conv2D(16, (5, 5), padding='same',
                                       activation='relu')(stn_input_layer)
        locnet_y = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(locnet_y)
        locnet_y = keras.layers.Flatten()(locnet_y)
        locnet_y = keras.layers.Dense(64, activation='relu')(locnet_y)
        locnet_y = keras.layers.Dense(6,
                                      weights=[
                                          np.zeros((64, 6), dtype='float32'),
                                          np.float32([[1, 0, 0], [0, 1, 0]]).flatten()
                                      ])(locnet_y)
        localization_net = keras.models.Model(inputs=stn_input_layer, outputs=locnet_y)
        x = keras.layers.Lambda(_transform,
                                output_shape=stn_input_output_shape)([x, localization_net(x)])
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
    backbone = keras.models.Model(inputs=inputs, outputs=x)
    x = keras.layers.Dropout(dropout, name='dropout')(x)
    x = keras.layers.Dense(len(alphabet) + 1,
                           kernel_initializer='he_normal',
                           activation='softmax',
                           name='fc_12')(x)
    x = keras.layers.Lambda(lambda x: x[:, rnn_steps_to_discard:])(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    prediction_model = keras.models.Model(inputs=inputs, outputs=CTCDecoder()(model.output))
    labels = keras.layers.Input(name='labels', shape=[model.output_shape[1]], dtype='float32')
    label_length = keras.layers.Input(shape=[1])
    input_length = keras.layers.Input(shape=[1])
    loss = keras.layers.Lambda(lambda inputs: keras.backend.ctc_batch_cost(
        y_true=inputs[0], y_pred=inputs[1], input_length=inputs[2], label_length=inputs[3]))(
            [labels, model.output, input_length, label_length])
    training_model = keras.models.Model(inputs=[model.input, labels, input_length, label_length],
                                        outputs=loss)
    training_model.compile(loss=lambda _, y_pred: y_pred, optimizer=optimizer)
    return backbone, model, training_model, prediction_model


class Recognizer:
    def __init__(self,
                 alphabet=string.digits + string.ascii_lowercase,
                 height=31,
                 width=200,
                 color=False,
                 filters=None,
                 rnn_units=None,
                 dropout=0.25,
                 optimizer=None,
                 stn=True,
                 rnn_steps_to_discard=2,
                 weights='kurapan',
                 include_top=True):
        if filters is None:
            filters = [64, 128, 256, 256, 512, 512, 512]
        assert len(filters) == 7, '7 CNN filters must be provided.'
        if rnn_units is None:
            rnn_units = [128, 128]
        assert len(rnn_units) == 2, '2 RNN filters must be provided.'
        self.alphabet = alphabet
        self.blank_label_idx = len(alphabet)
        self.backbone, self.model, self.training_model, self.prediction_model = build_model(
            alphabet=alphabet,
            height=height,
            width=width,
            stn=stn,
            optimizer=optimizer,
            rnn_steps_to_discard=rnn_steps_to_discard,
            color=color,
            filters=filters,
            rnn_units=rnn_units,
            dropout=dropout)
        if weights is not None:
            pretrained_key = (weights, include_top, tuple(filters), tuple(rnn_units), color, stn)
            assert pretrained_key in PRETRAINED_WEIGHTS, (
                'No pretrained weights available for this configuration. '
                'Please set weights=None.')
            pretrained_config = PRETRAINED_WEIGHTS[pretrained_key]
            if include_top:
                pretrained_target = self.model
                assert pretrained_config['alphabet'] == alphabet, (
                    'Provided alphabet does not match pretrained alphabet. '
                    'Please use `alphabet={alphabet}` or `include_top=False`').format(
                        alphabet=alphabet)
            else:
                pretrained_target = self.backbone
            pretrained_target.load_weights(
                tools.download_and_verify(url=pretrained_config['url'],
                                          sha256=pretrained_config['sha256']))

    def get_batch_generator(self, image_generator, batch_size=8, lowercase=False):
        """
        Generate batches of training data.

        Args:
            image_generator: An image / sentence tuple generator. The images should
                be in color even if the OCR is setup to handle grayscale as they
                will be converted here.
            batch_size: How many images to generate at a time.
            lowercase: Whether to convert all characters to lowercase before
                encoding.
        """
        y = np.zeros((batch_size, 1))
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
            sentences = [sentence.strip() for _, sentence, _ in batch]
            if lowercase:
                sentences = [sentence.lower() for sentence in sentences]
            assert all(c in self.alphabet
                       for c in ''.join(sentences)), 'Found illegal characters in sentence.'
            assert all(sentences), 'Found a zero length sentence.'
            assert all(
                len(sentence) <= max_string_length
                for sentence in sentences), 'A sentence is longer than this model can predict.'
            label_length = np.array([len(sentence) for sentence in sentences])[:, np.newaxis]
            labels = np.array([[self.alphabet.index(c) for c in sentence] + [self.blank_label_idx] *
                               (max_string_length - len(sentence)) for sentence in sentences])
            input_length = np.ones((batch_size, 1)) * max_string_length
            yield (images, labels, input_length, label_length), y

    def recognize(self, image):
        """Recognize text from a single image.

        Args:
            image: A pre-cropped image containing characters
        """
        image = tools.read_and_fit(filepath_or_array=image,
                                   width=self.prediction_model.input_shape[2],
                                   height=self.prediction_model.input_shape[1],
                                   cval=0)
        if self.prediction_model.input_shape[-1] == 1:
            image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        image = image.astype('float32') / 255
        return ''.join([
            self.alphabet[idx] for idx in self.prediction_model.predict(image[np.newaxis])[0]
            if idx not in [self.blank_label_idx, -1]
        ])

    def recognize_from_boxes(self,
                             image,
                             boxes,
                             batch_size=5) -> typing.List[typing.Tuple[str, np.ndarray]]:
        """Recognize text from an image using a set of bounding boxes.

        Args:
            image: A pre-cropped image containing characters
            boxes: A list of boxes provided as four coordinates
            batch_size: The prediction batch size
        """
        crops = []
        image = tools.read(image)
        for box in boxes:
            crops.append(
                tools.warpBox(image=image,
                              box=box,
                              target_height=self.model.input_shape[1],
                              target_width=self.model.input_shape[2]))
        crops = np.array(
            [cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)[..., np.newaxis] for crop in crops])
        X = crops.astype('float32') / 255
        predictions = []
        for index in range(0, len(X), batch_size):
            y = self.prediction_model.predict(X[index:index + batch_size])
            predictions.extend([
                ''.join(
                    [self.alphabet[idx] for idx in row if idx not in [self.blank_label_idx, -1]])
                for row in y
            ])
        return list(zip(predictions, boxes))
