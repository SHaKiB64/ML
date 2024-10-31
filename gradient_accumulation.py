import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K

class AdamAccumulate(Optimizer):
    '''
    Adam optimizer with gradient accumulation. Accumulates gradients over multiple
    batches before performing an optimization step, allowing for larger synthetic batch sizes.

    Args:
      accum_iters: Number of batches to accumulate gradients before updating model weights.
    '''
    def __init__(self, learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 amsgrad=False, accum_iters=1, **kwargs):
        super(AdamAccumulate, self).__init__(name="AdamAccumulate", **kwargs)
        
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.accum_iters = tf.Variable(accum_iters, dtype=tf.float32, trainable=False)

        # State variables to store accumulated gradients and the number of steps.
        self.steps = tf.Variable(0, dtype=tf.int64, trainable=False)

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        """Override to accumulate gradients."""
        grads, vars = zip(*grads_and_vars)
        
        # Initialize accumulators if not already done.
        if not hasattr(self, "accum_grads"):
            self.accum_grads = [tf.Variable(tf.zeros_like(var), trainable=False) for var in vars]

        # Accumulate gradients.
        for acc, grad in zip(self.accum_grads, grads):
            acc.assign_add(grad / self.accum_iters)

        # Apply accumulated gradients once enough iterations have passed.
        if tf.math.equal((self.steps + 1) % self.accum_iters, 0):
            super(AdamAccumulate, self).apply_gradients(
                [(acc, var) for acc, var in zip(self.accum_grads, vars)]
            )
            # Reset accumulated gradients.
            for acc in self.accum_grads:
                acc.assign(tf.zeros_like(acc))

        # Increment the step counter.
        self.steps.assign_add(1)

    def get_config(self):
        """Returns the configuration of the optimizer."""
        config = super(AdamAccumulate, self).get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad,
            "accum_iters": int(self.accum_iters.numpy())
        })
        return config
