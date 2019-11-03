import typing
# pylint: disable=invalid-name,too-many-locals,too-many-arguments
import keras
import keras.layers
import keras.backend
import keras_applications
import numpy as np
import cv2

from . import tools


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
        input_length = keras.backend.ones(shape=(keras.backend.shape(y_pred)[0],
                                                 1)) * keras.backend.cast(
                                                     keras.backend.shape(y_pred)[1], 'float32')

        return keras.backend.ctc_batch_cost(y_true=y_true,
                                            y_pred=y_pred,
                                            input_length=input_length,
                                            label_length=label_length)

    return loss


def decode(predictions, alphabet):
    """Decode a set of CTC predictions into actual strings.

    Args:
        predictions: Array of shape [N, L] where N is the number
            of samples and L is that max string length. Each
            entry is the index in alphabet for the recognized character.
        alphabet: A string of characters to index into.
    """
    return [
        ''.join([alphabet[i] if i < len(alphabet) and i != -1 else '' for i in sequence])
        for sequence in predictions
    ]


def CTCDecoder(rnn_steps_to_discard=2):  # pylint: disable=invalid-name
    def decoder(y_pred):
        y_pred = y_pred[:, rnn_steps_to_discard:, :]
        sequence_length = keras.backend.cast(keras.backend.shape(y_pred)[1], 'float32')
        input_length = keras.backend.ones_like(y_pred[:, 0, 0]) * (sequence_length - rnn_steps_to_discard)
        return keras.backend.ctc_decode(y_pred, input_length, greedy=True)[0]

    return keras.layers.Lambda(decoder, name='decode')


def build_model(alphabet_length,
                width=128,
                height=64,
                pool_size=2,
                filters=16,
                kernel_size=(3, 3),
                time_dense_size=32,
                rnn_size=512,
                color=False,
                rnn_type='lstm',
                kernel_initializer='he_normal',
                activation='relu'):
    input_layer = keras.layers.Input(shape=(height, width, 1 if not color else 3), dtype='float32')
    x = keras.layers.Permute((2, 1, 3))(input_layer)
    x = keras.layers.Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            padding='same',
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            name='conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(x)
    x = keras.layers.Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            padding='same',
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            name='conv2')(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(x)
    x = keras.layers.Reshape(target_shape=(width // (pool_size**2),
                                           (height // (pool_size**2)) * filters),
                             name='reshape')(x)

    # Cut down input size going into RNN.
    x = keras.layers.Dense(time_dense_size, activation=activation, name='dense1')(x)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    if rnn_type == 'lstm':
        RNN = keras.layers.LSTM
    elif rnn_type == 'gru':
        RNN = keras.layers.GRU
    else:
        raise NotImplementedError
    rnn_1 = RNN(rnn_size, return_sequences=True, kernel_initializer=kernel_initializer,
                name='gru1')(x)
    rnn_1b = RNN(rnn_size,
                 return_sequences=True,
                 go_backwards=True,
                 kernel_initializer=kernel_initializer,
                 name='gru1_b')(x)
    rnn1_merged = keras.layers.Add()([rnn_1, rnn_1b])
    rnn_2 = RNN(rnn_size, return_sequences=True, kernel_initializer=kernel_initializer,
                name='gru2')(rnn1_merged)
    rnn_2b = RNN(rnn_size,
                 return_sequences=True,
                 go_backwards=True,
                 kernel_initializer=kernel_initializer,
                 name='gru2_b')(rnn1_merged)

    # Transforms RNN output to character activations.
    x = keras.layers.Concatenate()([rnn_2, rnn_2b])
    x = keras.layers.Dense(alphabet_length + 1,
                           kernel_initializer=kernel_initializer,
                           name='dense2')(x)
    x = keras.layers.Activation('softmax', name='softmax')(x)
    return keras.models.Model(inputs=input_layer, outputs=x)


def build_prediction_model(model, rnn_steps_to_discard=2):
    return keras.models.Model(inputs=model.inputs,
                              outputs=CTCDecoder(rnn_steps_to_discard=rnn_steps_to_discard)(model.outputs[0]))


def build_training_model(model, max_string_length=8, rnn_steps_to_discard=2):
    labels = keras.layers.Input(name='labels', shape=[max_string_length], dtype='float32')
    loss = keras.layers.Lambda(make_ctc_loss(rnn_steps_to_discard=rnn_steps_to_discard),
                               output_shape=(1, ),
                               name='ctc')([model.outputs[0], labels])
    training_model = keras.models.Model(inputs=[model.inputs[0], labels], outputs=loss)
    training_model.compile(loss=lambda _, y_pred: y_pred, optimizer=keras.optimizers.RMSprop())
    return training_model


class Recognizer:
    """An OCR based on the Keras example (see
    https://github.com/keras-team/keras/blob/master/examples/image_ocr.py).
    The main difference is we use RMSprop instead of SGD and we change the calculation
    of the loss function so that only the labels are needed as input.

    Args:
        alphabet: A string of letters representing the characters
            the recognizer will detect.
        width: The width of the input images
        height: The height of the input images
        pool_size: The amount of pooling to be used after convolutional layers
        filters: The number of convolutional filters.
        kernel_size: The size of the convolutional filters
        rnn_size: The numer of recurrent units
        rnn_type: The type of RNN to use ('gru' or 'lstm')
        kernel_initializer: The Keras kernel initializer to use
        activation: The activation function.
        rnn_steps_to_discard: The number of RNN predictions to discard (the
            first few are usually not useful). This is used in both training and inference.
        preprocessing_mode: The preprocessing mode to use with
            keras_applications.imagenet_utils.preprocess_input
    """
    def __init__(self,
                 alphabet: str,
                 width=128,
                 height=64,
                 pool_size=2,
                 filters=16,
                 kernel_size=(3, 3),
                 time_dense_size=32,
                 rnn_size=512,
                 rnn_type='lstm',
                 kernel_initializer='he_normal',
                 activation='relu',
                 rnn_steps_to_discard=2,
                 preprocessing_mode='tf',
                 color=False):
        self.color = color
        self.rnn_steps_to_discard = rnn_steps_to_discard
        self.preprocessing_mode = preprocessing_mode
        self.alphabet = alphabet
        self.model = build_model(alphabet_length=len(alphabet),
                                 width=width,
                                 height=height,
                                 pool_size=pool_size,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 color=color,
                                 time_dense_size=time_dense_size,
                                 rnn_size=rnn_size,
                                 rnn_type=rnn_type,
                                 kernel_initializer=kernel_initializer,
                                 activation=activation)
        self.prediction_model = build_prediction_model(self.model,
                                                       rnn_steps_to_discard=rnn_steps_to_discard)
        self.training_model = None

    def create_training_model(self, max_string_length):
        """Build a training model.

        Args:
            max_string_length: The maximum length of string to
                use (for training only). For inference, the maximum string
                length is always equal to
                width // (pool_size ** 2) - rnn_steps_to_discard
        """
        if self.training_model is not None:
            raise Exception('A training model already exists.')
        self.training_model = build_training_model(self.model,
                                                   max_string_length=max_string_length,
                                                   rnn_steps_to_discard=self.rnn_steps_to_discard)

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
        blank_label = len(self.alphabet)
        if self.training_model is None:
            raise Exception('You must first call create_training_model().')
        max_string_length = keras.backend.int_shape(self.training_model.inputs[1])[-1]

        while True:
            batch = [data for data, _ in zip(image_generator, range(batch_size))]
            if not self.color:
                images = [
                    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
                    for image, _, _ in batch
                ]
            else:
                images = [image for image, _, _ in batch]
            images = np.array([image.astype('float32') for image in images])
            images = keras_applications.imagenet_utils.preprocess_input(
                images, mode=self.preprocessing_mode, data_format='channels_last', backend=keras.backend)
            sentences = [sentence for _, sentence, _ in batch]
            assert all(c in self.alphabet for c in ''.join(sentences)), 'Found illegal characters in sentence.'
            assert all(len(sentence) <= max_string_length for sentence in sentences)
            labels = np.array([[self.alphabet.index(c) for c in sentence] + [blank_label] *
                               (max_string_length - len(sentence)) for sentence in sentences])
            yield [images, labels], y

    def recognize(self, images: typing.List[typing.Union[np.ndarray, str]]) -> typing.List[str]:
        """Recognize the text in a set of images.

        Args:
            images: Can be a list of numpy arrays of shape HxWxC where C is
                1 if the image is grayscale and three if it is in color or
                a list of filepaths.
        """
        height, width = self.model.input_shape[1:-1]
        images = [tools.read_and_fit(image, width=width, height=height) for image in images]
        for index, image in enumerate(images):
            # It's a grayscale image with no
            # channel dimension so we need to add a
            # dimension first or possibly convert to color.
            if len(image.shape) == 2 and not self.color:
                images[index] = image[..., np.newaxis]
                continue
            # It's grayscale but we need color.
            if len(image.shape) == 2 and self.color:
                images[index] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                continue

            # It's a color image but we need grayscale
            if image.shape[2] == 3 and not self.color:
                images[index] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
                continue

            # It's a grayscale image with a color dimension
            # but we need color.
            if image.shape[2] == 1 and self.color:
                images[index] = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)

        images = np.array(images)
        X = keras_applications.imagenet_utils.preprocess_input(images,
                                                               mode=self.preprocessing_mode,
                                                               data_format='channels_last',
                                                               backend=keras.backend)
        predictions = self.prediction_model.predict(X)
        return decode(predictions=predictions, alphabet=self.alphabet)

    def recognize_from_boxes(self, image, boxes,
                             batch_size=5) -> typing.List[typing.Tuple[str, np.ndarray]]:
        crops = []
        for box in boxes:
            crops.append(
                tools.warpBox(image=image,
                              box=box,
                              target_height=self.model.input_shape[1],
                              target_width=self.model.input_shape[2]))
        crops = np.array(
            [cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)[..., np.newaxis] for crop in crops])
        X = keras_applications.imagenet_utils.preprocess_input(crops,
                                                               mode=self.preprocessing_mode,
                                                               data_format='channels_last',
                                                               backend=keras.backend)
        predictions = []

        for index in range(0, len(crops), batch_size):
            predictions.extend(
                decode(predictions=self.prediction_model.predict(X[index:index + batch_size]),
                       alphabet=self.alphabet))
        return list(zip(predictions, boxes))
