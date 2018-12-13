#  ====================================================
#   Filename: distribute.py
#   Author: Botao Xiao
#   Function: This is the entrance of the distributed training system.
#   We run the training program by calling this file.
#  ====================================================
import os
import sys

import tensorflow as tf

# ############################################################################################
# ################All modules are reserved for reflection, please don't modify imports######################
# ############################################################################################
import distribute_flags as flags
import distribute_train as Train
import distribute_annotations as annotations
import distribute_model as model
import distribute_input as Input
import distribute_eval as Eval


@annotations.current_model(model='MyModel')
@annotations.current_mode(mode='Train')
@annotations.current_input(input='MyDataLoader')
@annotations.current_feature( features={
            'hazed_image_raw': tf.FixedLenFeature([], tf.string),
            'clear_image_raw': tf.FixedLenFeature([], tf.string),
            'hazed_height': tf.FixedLenFeature([], tf.int64),
            'hazed_width': tf.FixedLenFeature([], tf.int64),
        })
@annotations.gpu_num(gpu_num=4)
@annotations.ps_hosts(ps_host="127.0.0.1: 22")
@annotations.worker_hosts(worker_hosts="127.0.0.1:23, 127.0.0.2: 24")
@annotations.job_name(job_name=flags.FLAGS.job_name)
@annotations.task_index(task_index=flags.FLAGS.task_index)
@annotations.batch_size(batch_size=35)
@annotations.sample_number(sample_number=100000000000000)
@annotations.epoch_num(epoch_num=100)
@annotations.model_dir(model_dir="")
def main():
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # ######################################################################
    # #######################Work on the annotations###########################
    # ######################################################################
    # Step 1: Get the model class and create a model
    experiment_model = annotations.get_instance_from_annotation(main, 'model', model)
    # Step 2: Get the mode, either train or eval
    mode = annotations.get_value_from_annotation(main, 'mode')
    if mode != 'Train' and mode != 'Eval':
        raise ValueError("mode must be set in the annotation @current_mode")
    # Step 3: Get Data loader for processing
    data_loader = annotations.get_instance_from_annotation(main, 'input', Input)
    if data_loader.type == "TFRecordDataLoader":
        if not hasattr(main, 'features'):
            raise ValueError("Please user @current_feature to create your features for data_loader")
        features = annotations.get_value_from_annotation(main, 'features')
        setattr(data_loader, 'features', features)
        input_mode = Input.InputOptions.TF_RECORD
    else:
        input_mode = Input.InputOptions.PLACEHOLDER
    # Step 4: Get traing or evaluation parameters
    gpu_num = annotations.get_value_from_annotation(main, "gpu_num")
    if gpu_num > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if gpu_num < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    batch_size = annotations.get_value_from_annotation(main, 'batch')
    epoch_num = annotations.get_value_from_annotation(main, 'epoch_num')
    sample_number = annotations.get_value_from_annotation(main, 'sample_num')
    setattr(data_loader, 'batch_size', batch_size)
    model_dir = annotations.get_value_from_annotation(main, "model_dir")
    if not os.path.exists(model_dir):
        raise ValueError("Path to save or restore model doesn't exist")
    # Step 5: Get training or evaluation instance and run.
    mod = sys.modules['__main__']
    operator_module = getattr(mod, mode)
    class_obj = getattr(operator_module, mod)
    operator = class_obj.__new__(class_obj)
    setattr(operator, 'data_loader', data_loader)
    setattr(operator, 'input_mode', input_mode)
    setattr(operator, 'batch_size', batch_size)
    setattr(operator, 'epoch_num', epoch_num)
    setattr(operator, 'sample_number', sample_number)
    setattr(operator, 'model_dir', model_dir)
    # Step 6: Get distribute specification
    ps_hosts = annotations.get_value_from_annotation(main, "ps_hosts")
    worker_hosts = annotations.get_value_from_annotation(main, "worker_hosts")
    ps_spec = ps_hosts.split(",")
    worker_spec = worker_hosts.split(",")
    job_name = annotations.get_value_from_annotation(main, "job_name")
    task_index = annotations.get_value_from_annotation(main, "task_index")
    setattr(operator, 'task_index', task_index)
    setattr(operator, 'job_name', job_name)
    cluster = tf.train.ClusterSpec({
            "ps": ps_spec,
            "worker": worker_spec})
    cluster.num_tasks("worker")
    setattr(operator, 'cluster', cluster)
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
    setattr(operator, 'server', server)
    operator.run()


if __name__ == '__main__':
    tf.app.run()
