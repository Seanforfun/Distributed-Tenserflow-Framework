#  ====================================================
#   Filename: distribute_estimator.py
#   Author: Botao Xiao
#   Function: This is the file that we create our own estimator.
#   I will encapsulate some functions so this file will be hidden for
#   most of the users.
#  ====================================================

import tensorflow as tf

from distribute_input import Input


def current_input(**kwds):
    def decorate(f):
        for k in kwds:
            if k == 'input':
                setattr(f, k, kwds[k])
        return f

    return decorate


@current_input(input="")
class DistributeEstimator(tf.estimator.Estimator):
    def __init__(self, model_fn, eval_model_fn=None, config=None, params=None, input_class=None):
        super().__init__(model_fn=model_fn, config=config, params=params)
        if input_class is not None:
            self.input_class = input_class
        else:
            model_classname = getattr(DistributeEstimator, 'input', 0)
            self.input_class = getattr(Input, model_classname)
        assert DistributeEstimator.input_class is not None, \
            "Please either pass your input class or use annotation @current_input"
        self.eval_model_fn = eval_model_fn


if __name__ == '__main__':
    pass
