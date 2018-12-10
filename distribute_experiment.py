#  ====================================================
#   Filename: distribute_experiment.py
#   Author: Botao Xiao
#   Function: This file is used to save the class of DistributeExperiment
#  ====================================================
import functools
import tensorflow as tf

import distribute_flags as flags
import distribute_estimator as est
from distribute_train import Train


class DistributeExperiment(object):
    @staticmethod
    def get_experiment_fn(num_gpus, variable_strategy):
        def __experiment_fn(run_config, hparams):

            # Create a estimator object
            distibute_estimator = est.DistributeEstimator(
                model_fn=Train.get_train_fn(num_gpus, variable_strategy,
                                            run_config.num_worker_replicas or 1, hparams),
                eval_model_fn=None,
                config=run_config,
                params=hparams,
                input_class=None
            )

            train_input_fn = functools.partial(
                distibute_estimator.input_class.load_train_batch,
                data_dir=flags.FLAGS.data_dir,
                batch_size=flags.FLAGS.batch_size,
                gpu_num=num_gpus
            )

            eval_input_fn = functools.partial(
                distibute_estimator.input_class.load_eval_batch,
                data_dir=flags.FLAGS.eval_data_dir,
                batch_size=flags.FLAGS.eval_batch_size,
                gpu_num=num_gpus
            )

            train_steps = flags.FLAGS.train_step
            evaluation_steps = flags.FLAGS.eval_example_num // flags.FLAGS.eval_batch_size

            # Create an experiment
            return tf.contrib.learn.Experiment(
                estimater=distibute_estimator,
                train_input_fn=train_input_fn,
                eval_input_fn=eval_input_fn,
                train_steps=train_steps,
                eval_steps=evaluation_steps,
                params=hparams)

        return __experiment_fn


if __name__ == '__main__':
    pass