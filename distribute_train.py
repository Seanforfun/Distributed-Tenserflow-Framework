#  ====================================================
#   Filename: distribute_train.py
#   Author: Botao Xiao
#   Function: The training file is used to save the training process
#  ====================================================

import multiprocessing

import tensorflow as tf

import distribute_constants as constant
import distribute_flags as flags
import distribute_net as net
import distribute_tower as tower
import distribute_utils as utils
from distribute_loss import Loss


class Train():
    @staticmethod
    def __create_done_queue(num_workers):
        with tf.device("/job:ps/task:0"):
            return tf.FIFOQueue(num_workers, tf.int32, shared_name="done_queue0")

    @staticmethod
    def get_train_fn(gpu_num, variable_strategy, num_workers, params):
        def _get_train_fn(raw_data, ground_truth, mode, params):
            cpu_num = multiprocessing.cpu_count()
            if gpu_num == 0:
                num_devices = 1
                device_type = 'cpu'
            else:
                num_devices = gpu_num
                device_type = 'gpu'

            ps_spec = flags.FLAGS.ps_hosts.split(",")
            worker_spec = flags.FLAGS.worker_hosts.split(",")
            num_workers = len(worker_spec)
            cluster = tf.train.ClusterSpec({
                "ps": ps_spec,
                "worker": worker_spec})

            kill_ps_queue = Train.__create_done_queue(num_workers)

            server = tf.train.Server(cluster, job_name=flags.FLAGS.job_name, task_index=flags.FLAGS.task_index)
            # ####################################################################################
            # #################################Parameter Server#####################################
            # ####################################################################################
            if flags.FLAGS.job_name == "ps":
                with tf.Session(server.target) as sess:
                    for i in range(num_workers):
                        sess.run(kill_ps_queue.dequeue())
                return

            # ####################################################################################
            # #################################Worker Service######################################
            # ####################################################################################
            is_chief = (flags.FLAGS.task_index == 0)
            worker_device = "/job:worker/task:%d" % flags.FLAGS.task_index
            ps_device = "/job:ps/cpu:0"
            for i in range(1, cpu_num):
                ps_device += ", /job:ps/cpu: %d" % i
            with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device=ps_device,
                                                          cluster=cluster)):
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                              trainable=False)
                tower_grads = []
                tower_losses = []
                optimizer = tf.train.AdamOptimizer(constant.INITIAL_LEARNING_RATE)
                with tf.variable_scope(tf.get_variable_scope()):
                    for i in range(num_devices):
                        with tf.device('/%s:%d' % (device_type, i)):
                            with tf.name_scope('%s_%d' % (constant.TOWER_NAME, i)) as scope:
                                current_net = net.Net()
                                current_tower = tower.Tower(current_net, scope, tower_grads, raw_data, ground_truth, Loss.loss_fn, optimizer)
                                summaries, loss = current_tower.process()
                                tower_losses.append(loss)

                # We must calculate the mean of each gradient. Note that this is the
                # synchronization point across all towers.
                grads = tower.Tower.average_gradients(tower_grads)

                loss = tf.reduce_mean(tower_losses, name='loss')
                examples_sec_hook = utils.ExamplesPerSecondHook(
                    flags.FLAGS.batch_size, every_n_steps=10)
                tensors_to_log = {'loss': loss}
                logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=100)
                train_hooks = [logging_hook, examples_sec_hook]

                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, use_locking=False,
                    replicas_to_aggregate=num_workers,
                    total_num_replicas=num_workers,
                    name="sync_replicas")
                sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
                train_hooks.append(sync_replicas_hook)

                # Apply the gradients to adjust the shared variables
                train_op = optimizer.apply_gradients(grads, global_step=global_step)
                train_op = tf.group(*train_op)

                return tf.estimator.EstimatorSpec(
                    mode="train",
                    loss=loss,
                    train_op=train_op,
                    training_hooks=train_hooks)

        return _get_train_fn


if __name__ == '__main__':
    pass
