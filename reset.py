# this is rest.py

import tensorflow as tf
import gc

def reset_keras():
    '''
    Perform garbage collection and release GPU memory in TensorFlow 2.x.
    '''
    # Clear any Keras/TensorFlow states
    tf.keras.backend.clear_session()

    # Run garbage collection to free memory
    print(gc.collect())  # If memory is freed, you should see a number printed.

    # Configure GPU memory management
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Limit TensorFlow to use only the first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Enable memory growth to prevent TensorFlow from reserving all GPU memory at startup
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(f"Error configuring GPU memory: {e}")

# Example usage
if __name__ == "__main__":
    reset_keras()

