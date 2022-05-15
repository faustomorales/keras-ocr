import os
import tensorflow as tf


def configure():
    memory_growth = os.environ.get("MEMORY_GROWTH", False)
    memory_allocated = os.environ.get("MEMORY_ALLOCATED", False)

    if memory_growth:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                raise e
        else:
            print("Memory growth set but no GPUs detected")

    if memory_allocated and isinstance(memory_allocated, float):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = (  # pylint: disable=no-member
            memory_allocated
        )
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
