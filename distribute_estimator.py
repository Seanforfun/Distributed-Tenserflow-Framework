#  ====================================================
#   Filename: distribute_estimator.py
#   Author: Botao Xiao
#   Function: This is the file that we create our own estimator.
#   I will encapsulate some functions so this file will be hidden for
#   most of the users.
#  ====================================================

import functools
import tensorflow as tf

from distribute_input import Input
from distribute_train import Train
import distribute_flags as flags


class DistributeEstimator(tf.estimator.Estimator):
    def __init__(self, model_fn):
        super().__init__(model_fn)

    @staticmethod
    def get_experiment_fn(num_gpus, variable_strategy, hparams):
        def __experiment_fn(run_config):
            train_input_fn = functools.partial(
            Input.load_data,
            flags.FLAGS.data_dir,
            num_gpus)

            train_steps = flags.FLAGS.train_step

            trainer = tf.estimator.Estimator(
                model_fn=Train.get_train_fn(num_gpus, variable_strategy,
                                      run_config.num_worker_replicas or 1),
                config=run_config,
                params=hparams)

            # Create an experiment
            return tf.contrib.learn.Experiment(
                trainer,
                train_input_fn=train_input_fn,
                train_steps=train_steps,
                params=hparams)

        return __experiment_fn


if __name__ == '__main__':
    pass
