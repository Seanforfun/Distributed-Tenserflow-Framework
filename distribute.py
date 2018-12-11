#  ====================================================
#   Filename: distribute.py
#   Author: Botao Xiao
#   Function: This is the entrance of the distributed training system.
#   We run the training program by calling this file.
#  ====================================================
import os

import tensorflow as tf

import distribute_experiment as experiment
import distribute_flags as flags
import distribute_train as train


def main(self):
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    mode = flags.FLAGS.mode
    gpu_num = flags.FLAGS.gpu_num

    if gpu_num > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if gpu_num < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')

    operation = experiment.DistributeExperiment(mode, train_fn=train.Train.train)
    operation.run()


if __name__ == '__main__':
    tf.app.run()
