#  ====================================================
#   Filename: distribute_flags.py
#   Author: Botao Xiao
#   Function: This is file used to save the flags we can call them
#   using command line.
#  ====================================================
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Distributed training options
tf.app.flags.DEFINE_string('ps_hosts', './DeHazeNetModel/trainLearningRate.json',
                           """Path to save training learning rate json file.""")
tf.app.flags.DEFINE_string('worker_hosts', './DeHazeNetModel/trainLearningRate.json',
                           """Path to save training learning rate json file.""")
tf.app.flags.DEFINE_integer('intra_op_parallelism_threads ', 0,
                            """
                              Number of threads to use for intra-op parallelism. When training on CPU
                              set to 0 to have the system pick the appropriate number or alternatively
                              set it to the number of physical CPU cores.
                            """)
tf.app.flags.DEFINE_integer('inter_op_parallelism_threads ', 0,
                            """
                             Number of threads to use for inter-op parallelism. If set to 0, the
                            system will pick an appropriate number.
                            """)
tf.app.flags.DEFINE_integer('train_step', 10000,
                           """
                           Number of steps to run for current worker.
                           """)
tf.app.flags.DEFINE_boolean('sync', False,
                            """Whether to have all workers update synchronize""")

# Training options
tf.app.flags.DEFINE_integer('data_load_option', 1,
                            """
                            1: Using tf-record to load data(Preferred).
                            2: Use placeholder to create data.
                            """)
tf.app.flags.DEFINE_integer('gpu_num', 1,
                            """
                            The number of gpus used. Uses only CPU if set to 0.
                            """)
tf.app.flags.DEFINE_string('variable_strategy', 'CPU',
                           """
                           Where to locate variable operations
                           CPU: locate variables on CPU
                           GPU: locate variables on GPU
                           """)
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                           """
                           Whether to log device placement.
                           """)

# Training parameters
tf.app.flags.DEFINE_integer('batch_size', 35,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('input_image_height', 224,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('input_image_width', 224,
                            """Input image width.""")
tf.app.flags.DEFINE_integer('sample_number', 100000,
                            """Total sample numbers to train.""")

# Files position
tf.app.flags.DEFINE_string('train_learning_rate', './DeHazeNetModel/trainLearningRate.json',
                           """Path to save training learning rate json file.""")

# Model position
tf.app.flags.DEFINE_string('model_position', '',
                           """Path to save training model.""")


if __name__ == '__main__':
    pass