#  ====================================================
#   Filename: distribute_experiment.py
#   Author: Botao Xiao
#   Function: This file is used to save the class of DistributeExperiment
#  ====================================================
import functools

import tensorflow as tf

import distribute_estimator as est
import distribute_flags as flags
import distribute_input as Input
from distribute_train import Train


def current_input(**kwds):
    def decorate(f):
        for k in kwds:
            if k == 'train_input' or k == 'eval_input':
                setattr(f, k, kwds[k])
        return f
    return decorate


@current_input(train_input="", eval_input='')
class DistributeExperiment(object):
    def __init__(self, mode, train_fn=None, train_input_fn=None, eval_fn=None, eval_input_fn=None):
        self.mode = mode
        if train_fn is None and eval_fn is None:
            raise ValueError("At least provide a funtion for processing")
        if mode == "Train":
            if train_fn is None:
                raise ValueError("In Train mode, train_fn cannot be None")
            else:
                self.train_fn = train_fn
            if train_input_fn is None:
                if hasattr(DistributeExperiment, 'train_input'):
                    model_classname = getattr(Input, 'input', 0)
                    train_input_fn_class = getattr(Input, model_classname)
                    self.train_input_fn = train_input_fn_class.load_train_batch
                else:
                    raise ValueError("In Train mode, train_input_fn must be provided.")
            else:
                self.train_input_fn = train_input_fn
        elif mode == "Eval":
            if eval_fn is None:
                raise ValueError("In Eval mode, eval_fn must be provided.")
            else:
                self.eval_fn = eval_fn
            if eval_input_fn is None:
                if hasattr(DistributeExperiment, 'eval_input'):
                    model_classname = getattr(Input, 'input', 0)
                    eval_input_fn_class = getattr(Input, model_classname)
                    self.eval_input_fn = eval_input_fn_class.load_eval_batch
                else:
                    raise ValueError("In Eval mode, eval_input_fn must be provided.")
            else:
                self.eval_input_fn = eval_input_fn
        else:
            raise ValueError("Please provide either Train or Eval as mode.")

    def train(self, pre_train_fn=None, post_train_fn=None, param=None):
        self.train_fn(self.train_input_fn, pre_train_fn, post_train_fn, param)



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

            train_steps = flags.FLAGS.batch_per_epoch * flags.FLAGS.batch_per_epoch
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