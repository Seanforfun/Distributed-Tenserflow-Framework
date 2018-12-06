#  ====================================================
#   Filename: distribute_train.py
#   Author: Botao Xiao
#   Function: The training file is used to save the training process
#  ====================================================

import inspect
import tensorflow as tf

import distribute_log as logger
import distribute_model as model


def train(self):
    """Returns an Experiment function.

    Experiments perform training on several workers in parallel,
    in other words experiments know how to invoke train and eval in a sensible
    fashion for distributed training. Arguments passed directly to this
    function are not tunable, all other arguments should be passed within
    tf.HParams, passed to the enclosed function.

    Args:
    Returns:
        A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
        tf.contrib.learn.Experiment.

        Suitable for use by tf.contrib.learn.learn_runner, which will run various
        methods on Experiment (train, evaluate) based on information
        about the current runner in `run_config`.
    """
    logger.info("Start training process.")
    tf.reset_default_graph()
    with tf.Graph().as_default():
        pass


if __name__ == '__main__':
    pass
