#  ====================================================
#   Filename: distribute_train.py
#   Author: Botao Xiao
#   Function: The training file is used to save the training process
#  ====================================================

import inspect
import tensorflow as tf

import distribute_flags as flags
import distribute_log as logger
import distribute_utils as utils
import distribute_model as model
import distribute_tower as tower
import distribute_net as net
import distribute_learningrate as learning_rate
import distribute_constants as constant


class Train():
    @staticmethod
    def get_train_fn(gpu_num, variable_strategy, worker_num):
        def _get_train_fn(raw_data, ground_truth):
            if gpu_num == 0:
                num_devices = 1
                device_type = 'cpu'
            else:
                num_devices = gpu_num
                device_type = 'gpu'

            initial_learning_rate = learning_rate.LearningRate(constant.INITIAL_LEARNING_RATE,
                                                               flags.FLAGS.learning_rate_json)
            # Create an optimizer that performs gradient descent.
            optimizer = tf.train.AdamOptimizer(initial_learning_rate.load())
            tower_grades = []
            for i in range(num_devices):
                worker_device = '/{}:{}'.format(device_type, i)
                if variable_strategy == 'CPU':
                    device_setter = utils.local_device_setter(
                        worker_device=worker_device)
                elif variable_strategy == 'GPU':
                    device_setter = utils.local_device_setter(
                        ps_device_type='gpu',
                        worker_device=worker_device,
                        ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                            gpu_num, tf.contrib.training.byte_size_load_fn))
                with tf.variable_scope(flags.FLAGS.project_name, reuse=bool(i != 0)):
                    with tf.name_scope('%s_%d' % (constant.TOWER_NAME, i)) as scope:
                        with tf.device(device_setter):
                            net, scope, tower_grades, optimizer, raw_data, ground_truth, loss_fn
                            current_net = net.Net()
                            current_tower = tower.Tower(current_net, scope, tower_grades, optimizer, raw_data, ground_truth, )
                            tower.Tower.tower_fn()

        return _get_train_fn


def train():
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
