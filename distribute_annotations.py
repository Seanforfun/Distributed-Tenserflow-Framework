#  ====================================================
#   Filename: distribute_annotations.py
#   Author: Botao Xiao
#   Function: This file is used to save all annotations for reflection.
#  ====================================================


def current_model(**kwds):
    """
    User  @current_model annotation to define current CNN model.
    This model must be a class in distribute_model.py implements
    interface Model.
    This annotation is call at main function.
    :param kwds: Create a dict {model: '#Fill with your model class name#'}
    :return:  decorate
    :example: @current_model(model="MyModel")
    """
    def decorate(f):
        for k in kwds:
            if k == 'model':
                setattr(f, k, kwds[k])
        return f
    return decorate


def current_input(**kwds):
    """
    Annotation for creating data loader instance.
    The custom class must be defined in distribute_input.py.
    1. If user is using tf-record as input, please implement TFRecordDataLoader class.
    2. If user is using placeholder as input, please implement PlaceholderDataLoader class.
    :param kwds: Dict, user give the name of your customized class name with key 'input'.
    :return: decorate
    :example: @current_input(input='MyDataLoader')
    """
    def decorate(f):
        for k in kwds:
            if k == 'input':
                setattr(f, k, kwds[k])
        return f
    return decorate


def current_mode(**kwds):
    """
    Annotation for getting mode for current experiment.
    Use 'Train' as value for training.
    Use 'Eval' as value for evaluation.
    :param kwds: Dict, user give the name of your customized class name with key 'mode'.
    :return: decorate
    :example: @current_mode(mode='Train')
    """
    def decorate(f):
        for k in kwds:
            if k == 'mode':
                setattr(f, k, kwds[k])
        return f
    return decorate


def current_feature(**kwds):
    """
    Annotation for getting feature for current tf-record dataloader.
    :param kwds: Dict, user give the name of your customized class name with key 'features'.
    :return: decorate
    :example: @current_feature( features={
            'hazed_image_raw': tf.FixedLenFeature([], tf.string),
            'clear_image_raw': tf.FixedLenFeature([], tf.string),
            'hazed_height': tf.FixedLenFeature([], tf.int64),
            'hazed_width': tf.FixedLenFeature([], tf.int64),
        })
    """
    def decorate(f):
        for k in kwds:
            if k == 'feature':
                setattr(f, k, kwds[k])
        return f
    return decorate


def get_advice(**kwds):
    """
    Get advice and inject the advices to specific aspects,
    which are:
    1. pre_fn: (Optional) A handler executed at the beginning.
    2. post_fn: (Optional) A handler executed at the end.
    3. pre_process_fn: (Optional) A handler after getting data and before
    going into the net.
    4. post_processs_fn: (Optional)  A handler after getting the results from
    the net and before lsos calculation.
    :scope: Put it on the Train class in distribute_train.py for training or
    Eval class in distribute_eval.py for evaluation.
    :param kwds: Dict, we can use it to pass function handlers, keys are 'pre_fn',
    'post_fn', 'pre_process_fn' and 'post_processs_fn'.
    :return:decorate
    example: @get_device(pre_fn=handler1, post_fn=handler2, pre_process_fn=handler3,
    post_process_fn=handler4)
    """
    def decorate(f):
        for k in kwds:
            if k == 'pre_fn' or k == 'post_fn' or k == 'pre_process_fn' or k == 'post_processs_fn':
                setattr(f, k, kwds[k])
        return f
    return decorate


def gpu_num(**kwds):
    """
    Annotation for getting number using gpu.
    :param kwds: Dict, user give the name of your customized class name with key 'gpu_num'.
    :return: decorate
    :example: @current_feature(gpu_num = 4)
    """
    def decorate(f):
        for k in kwds:
            if k == 'gpu_num':
                setattr(f, k, kwds[k])
        return f
    return decorate


def get_instance_from_annotation(current_obj, attr_name, module):
    """
    Create instance using annotations
    :param current_obj: Current object.
    :param attr_name:  Attribute name.
    :param module: The module to find the class.
    :return: A instance with all information settle done.
    """
    if not hasattr(current_obj, attr_name):
        raise ValueError("Current instance doesn't have this attribute.")
    model_class_name = getattr(current_obj, attr_name, 0)
    model_class = getattr(module, model_class_name)
    instance = model_class.__new__(model_class)
    if instance is None:
        raise ValueError("Cannot create instance with current annotaion.")
    return instance


def get_value_from_annotation(current_obj, attr_name):
    if not hasattr(current_obj, attr_name):
        raise ValueError("Current instance doesn't have this attribute.")
    return getattr(current_obj, attr_name)


if __name__ == '__main__':
    pass
