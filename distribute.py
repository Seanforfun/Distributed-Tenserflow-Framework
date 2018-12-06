#  ====================================================
#   Filename: distribute.py
#   Author: Botao Xiao
#   Function: This is the entrance of the distributed training system.
#   We run the training program by calling this file.
#  ====================================================
import os
import tensorflow as tf

import distribute_flags as flags
import distribute_estimator as estimator
import distribute_utils as utils
import distribute_train as train
import distribute_log as logger


def main(self):
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_num = flags.FLAGS.gpu_num
    variable_strategy = flags.FLAGS.variable_strategy
    batch_size = flags.FLAGS.batch_size
    log_device_placement = flags.FLAGS.log_device_placement
    num_intra_threads = flags.FLAGS.intra_op_parallelism_threads
    model_dir = flags.FLAGS.model_dir

    if gpu_num > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if gpu_num < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if gpu_num == 0 and variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if gpu_num != 0 and batch_size % gpu_num != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')

    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement,
        intra_op_parallelism_threads=num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = utils.RunConfig(
        session_config=sess_config, model_dir=model_dir)

    tf.contrib.learn.learn_runner.run(
        estimator.DistributeEstimator.get_experiment_fn(gpu_num, variable_strategy),
        run_config=config,
        hparams=tf.contrib.training.HParams(
            is_chief=config.is_chief))


if __name__ == '__main__':
    tf.app.run()
