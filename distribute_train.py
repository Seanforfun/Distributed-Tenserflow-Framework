#  ====================================================
#   Filename: distribute_train.py
#   Author: Botao Xiao
#   Function: The training file is used to save the training process
#  ====================================================

import six
import itertools
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
from distribute_loss import Loss
from distribute_tower import Tower


class Train():
    @staticmethod
    def get_train_fn(gpu_num, variable_strategy, num_workers, params):
        def _get_train_fn(raw_data, ground_truth, mode, params):
            if gpu_num == 0:
                num_devices = 1
                device_type = 'cpu'
            else:
                num_devices = gpu_num
                device_type = 'gpu'

            tower_losses = []
            tower_gradvars = []
            tower_preds = []
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
                            current_net = net.Net()
                            current_tower = tower.Tower(current_net, scope, tower_losses, tower_gradvars, tower_preds, raw_data, ground_truth, Loss.loss_fn)
                            loss, logist, gradvars = Tower.tower_fn(current_tower)
                            tower_losses.append(loss)
                            tower_gradvars.append(gradvars)
                            tower_preds.append(logist)
                            if i == 0:
                                # Only trigger batch_norm moving mean and variance update from
                                # the 1st tower. Ideally, we should grab the updates from all
                                # towers but these stats accumulate extremely fast so we can
                                # ignore the other stats from the other towers without
                                # significant detriment.
                                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                               scope)

                # Now compute global loss and gradients.
                gradvars = []
                with tf.name_scope('gradient_averaging'):
                    all_grads = {}
                    for grad, var in itertools.chain(*tower_gradvars):
                        if grad is not None:
                            all_grads.setdefault(var, []).append(grad)
                    for var, grads in six.iteritems(all_grads):
                        # Average gradients on the same device as the variables
                        # to which they apply.
                        with tf.device(var.device):
                            if len(grads) == 1:
                                avg_grad = grads[0]
                            else:
                                avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                        gradvars.append((avg_grad, var))

                # Device that runs the ops to apply global gradient updates.
                consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
                with tf.device(consolidation_device):
                    lr = constant.INITIAL_LEARNING_RATE
                    loss = tf.reduce_mean(tower_losses, name='loss')
                    examples_sec_hook = utils.ExamplesPerSecondHook(
                        flags.FLAGS.batch_size, every_n_steps=10)

                    tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

                    logging_hook = tf.train.LoggingTensorHook(
                        tensors=tensors_to_log, every_n_iter=100)
                    train_hooks = [logging_hook, examples_sec_hook]

                    optimizer = tf.train.AdamOptimizer(lr)

                    if flags.FLAGS.sync:
                        optimizer = tf.train.SyncReplicasOptimizer(
                            optimizer, replicas_to_aggregate=num_workers)
                        sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
                        train_hooks.append(sync_replicas_hook)

                    # Create single grouped train op
                    train_op = [
                        optimizer.apply_gradients(
                            gradvars, global_step=tf.train.get_global_step())
                    ]
                    train_op.extend(update_ops)
                    train_op = tf.group(*train_op)

                    predictions = {
                        # Create your data format using logist
                    }

                    return tf.estimator.EstimatorSpec(
                        mode="train",
                        predictions=predictions,
                        loss=loss,
                        train_op=train_op,
                        training_hooks=train_hooks)

        return _get_train_fn


if __name__ == '__main__':
    pass
