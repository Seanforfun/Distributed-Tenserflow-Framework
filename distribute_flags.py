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
tf.app.flags.DEFINE_string('mode', "Train",
                           """Train: for training; Eval: for evaluation""")


# Distributed training options
tf.app.flags.DEFINE_string('project_name', 'Your project name',
                           """String to save the project name.""")
tf.app.flags.DEFINE_string('job_name', '',
                           """One of ps and worker""")
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """The hosts that runs as parameter server.""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """The hosts that works as workers who are responsible for processing.""")
tf.app.flags.DEFINE_integer("task_index", None,
                            "Worker task index, should be >= 0. task_index=0 is "
                            "the master worker task the performs the variable "
                            "initialization ")
tf.app.flags.DEFINE_integer("replicas_to_aggregate", None,
                            "Number of replicas to aggregate before parameter update"
                            "is applied (For sync_replicas mode only; default: "
                            "num_workers)")


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
tf.app.flags.DEFINE_integer('gpu_num', 1,
                            """
                            The number of gpus used. Uses only CPU if set to 0.
                            """)
tf.app.flags.DEFINE_integer('epoch_num', 20,
                            """
                            Number of epoch for training.
                            """)
tf.app.flags.DEFINE_integer('batch_per_epoch', 111111111111111,
                            """
                            The number of batch in one epoch.
                            """)
tf.app.flags.DEFINE_string('variable_strategy', 'CPU',
                           """
                           Where to locate variable operations.
                           CPU: locate variables on CPU, CPU is like the parameter server.
                           GPU: locate variables on GPU, GPU is like the parameter server.
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
tf.app.flags.DEFINE_float('train_learning_rate', 0.001,
                          """Value of initial learning rate.""")

# Training Input parameters
tf.app.flags.DEFINE_string('data_dir', 'YOUR DATA PATH',
                           """Path to save data.""")
tf.app.flags.DEFINE_string('data_load_option', "tfrecords",
                           """
                            Select from either 'tfrecords' or 'placeholder'
                            'tfrecords': Using tf-record to load data(Preferred).
                            'placeholder': Use placeholder to create data.
                            """)

# Evaluation Input parameters
tf.app.flags.DEFINE_integer('eval_example_num', 10000,
                            """Number of evaluation examples.""")
tf.app.flags.DEFINE_string('eval_data_dir', 'YOUR EVALUATION DATA PATH',
                           """Path to save evaluation data.""")
tf.app.flags.DEFINE_integer('eval_batch_size', 1,
                            """Size of your evaluation batch, normally should be 1.""")

# Files position
tf.app.flags.DEFINE_string('learning_rate_json', 'YOUR LEARNING RATE SAVING PATH',
                           """Path to save learning rate json file.""")

# Model position
tf.app.flags.DEFINE_string('model_dir', '',
                           """Path to save training model.""")


if __name__ == '__main__':
    pass