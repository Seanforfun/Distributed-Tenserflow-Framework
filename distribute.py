#  ====================================================
#   Filename: distribute.py
#   Author: Botao Xiao
#   Function: This is the entrance of the distributed training system.
#   We run the training program by calling this file.
#  ====================================================
import os
import sys

import tensorflow as tf

import distribute_experiment as experiment
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
    input_mode = None
    if data_loader.type == "TFRecordDataLoader":
        if not hasattr(main, 'features'):
            raise ValueError("Please user @current_feature to create your features for data_loader")
        features = annotations.get_value_from_annotation(main, 'features')
        setattr(data_loader, 'features', features)
        input_mode = Input.InputOptions.TF_RECORD
    else:
        input_mode = Input.InputOptions.PLACEHOLDER
    # Steps 3: Get training or evaluation instance and run.
    mod = sys.modules['__main__']
    operator_module = getattr(mod, mode)
    class_obj = getattr(operator_module, mod)
    operator = class_obj.__new__(class_obj)
    setattr(operator, 'data_loader', data_loader)
    setattr(operator, 'input_mode', input_mode)
    operator.run()
    # Step 4: Get traing or evaluation gpu number
    gpu_num = annotations.get_value_from_annotation(main, "gpu_num")


    gpu_num = flags.FLAGS.gpu_num

    if gpu_num > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if gpu_num < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')

    operation = experiment.DistributeExperiment(mode, train_fn=Train.Train.train)
    operation.run()


if __name__ == '__main__':
    tf.app.run()
