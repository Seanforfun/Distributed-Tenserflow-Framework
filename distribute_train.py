#  ====================================================
#   Filename: distribute_train.py
#   Author: Botao Xiao
#   Function: The training file is used to save the training process
#  ====================================================

import multiprocessing
import time

import tensorflow as tf

import distribute_constants as constant
import distribute_flags as flags
import distribute_log as logger
import distribute_net as net
import distribute_tower as tower
from distribute_loss import Loss
import distribute_input as Input
from distribute_input import Dataloader


class Train(object):
    @staticmethod
    def __create_done_queue(num_workers):
        with tf.device("/job:ps/task:0"):
            return tf.FIFOQueue(num_workers, tf.int32, shared_name="done_queue0")

    @staticmethod
    def train(train_dataloader,
              input_mode,
              pre_train_fn=None,
              post_train_fn=None,
              pre_process_fn=None,
              post_process_fn=None,
              *args,
              **kwargs):
        """
        :param train_dataloader: A data loader used to get the data from data_dir.
        :param input_mode: One of using tf-records and placeholder. Please specify in distribute_flags.py or command.
        :param pre_train_fn: (Optional) A handler of pre train process.
        :param post_train_fn: (Optional) A handler of post train process.
        :param pre_process_fn: (Optional) A handler of process raw data and ground truth before pass them to the net.
        :param post_process_fn: (Optional) A handler of post the direct result from the net(before calculating loss)
        :param args: (Optional) User's additional parameters.
        :param kwargs: (Optional) User's additional dict.
        :return:
        """
        # ####################################################################
        # #####################Parameters Loading###############################
        # ####################################################################
        cpu_num = multiprocessing.cpu_count()
        gpu_num = flags.FLAGS.gpu_num
        task_index = flags.FLAGS.task_index
        is_chief = task_index == 0
        job_name = flags.FLAGS.job_name
        model_dir = flags.FLAGS.model_dir
        train_data_dir = flags.FLAGS.data_dir
        batch_size = flags.FLAGS.batch_size
        replicas_to_aggregate = flags.FLAGS.replicas_to_aggregate
        total_step = flags.FLAGS.epoch_num * flags.FLAGS.batch_per_epoch
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
        if job_name == "ps":
            with tf.Session(server.target) as sess:
                for i in range(num_workers):
                    sess.run(kill_ps_queue.dequeue())
            return

        # ####################################################################################
        # #################################Worker Service######################################
        # ####################################################################################
        worker_device = "/job:worker/task:%d" % flags.FLAGS.task_index
        ps_device = "/job:ps/cpu:0"
        for i in range(1, cpu_num):
            ps_device += ", /job:ps/cpu: %d" % i

        # ####################################################################################
        # #############################Pre Train Function ########################################
        # ####################################################################################
        pre_train_result = None
        if pre_train_fn is not None:
            pre_train_result = pre_train_fn(args, kwargs)

        if input_mode == Input.InputOptions.TF_RECORD:
            batch_queue = train_dataloader.load_queue_from_tfrecord(train_data_dir, batch_size)

        # ####################################################################################
        # #############################Training Function ########################################
        # ####################################################################################
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device=ps_device,
                                                          cluster=cluster)):
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                              trainable=False)
            tower_grads = []
            tower_losses = []
            tower_logist = []
            optimizer = tf.train.AdamOptimizer(constant.INITIAL_LEARNING_RATE)
            with tf.variable_scope(tf.get_variable_scope()):
                    for i in range(num_devices):
                        with tf.device('/%s:%d' % (device_type, i)):
                            with tf.name_scope('%s_%d' % (constant.TOWER_NAME, i)) as scope:
                                current_net = net.Net()
                                if input_mode == Input.InputOptions.TF_RECORD:
                                    raw_data, ground_truth = train_dataloader.load_train_batch(train_data_dir, batch_size, batch_queue=batch_queue)
                                else:
                                    raw_data, ground_truth = train_dataloader.load_train_batch
                                if pre_process_fn is not None:
                                        raw_data, ground_truth = pre_train_fn(raw_data, ground_truth, args, kwargs)
                                current_tower = tower.Tower(current_net, scope, tower_grads, raw_data, ground_truth, Loss.loss_fn, optimizer)
                                summaries, loss, logist = current_tower.process(post_process_fn, pre_train_result)
                                tower_losses.append(loss)
                                tower_logist.append(logist)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = tower.Tower.average_gradients(tower_grads)
            loss = tf.reduce_mean(tower_losses, name='loss')

            if replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = replicas_to_aggregate

            optimizer = tf.train.SyncReplicasOptimizer(
                optimizer, use_locking=False,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="sync_replicas")

            # Apply the gradients to adjust the shared variables
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

            chief_queue_runner = optimizer.get_chief_queue_runner()
            token_nums = max(replicas_to_aggregate - num_workers, 0)
            sync_init_op = optimizer.get_init_tokens_op(token_nums)

            init_op = tf.global_variables_initializer()
            kill_ps_enqueue_op = kill_ps_queue.enqueue(1)

            supervisor = tf.train.MonitoredTrainingSession(
                is_chief=is_chief,
                checkpoint_dir=model_dir,
                scaffold=tf.train.Scaffold(init_op),
                log_step_count_steps=1000
            )

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)

            if is_chief:
                logger.info("Worker %d: Initializing session..." % task_index)
            else:
                logger.info("Worker %d: Waiting for session to be initialized..." % task_index)
            sess = supervisor.prepare_or_wait_for_session(server.target, config=sess_config)

            logger.info("Worker %d: Session initialization complete." % task_index)
            if is_chief:
                supervisor.start_queue_runners(sess, [chief_queue_runner])
                sess.run(sync_init_op)

            start = time.time()
            while not supervisor.should_stop():
                _, step, loss_value = sess.run([train_op, global_step, loss])
                duration = time.time() - start

                if step % 10 == 0:
                    num_examples_per_step = batch_size * gpu_num
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = gpu_num
                    format_str = ('step %d, loss = %.8f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    logger.info(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                if step >= total_step:
                    break
            sess.run(kill_ps_enqueue_op)
            logger.info('kill_ps_enqueue_op done....')
            supervisor.stop()

        # ####################################################################################
        # #############################Post Train Function #######################################
        # ####################################################################################
        if post_train_fn is not None:
            post_train_fn(args, kwargs)


if __name__ == '__main__':
    pass
