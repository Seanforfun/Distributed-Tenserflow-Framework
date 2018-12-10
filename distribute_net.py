#  ====================================================
#   Filename: distribute_net.py
#   Function: This file is used to define the CNN network structure
#   actions, users can override pre_process and post_process to
#   add aspects for the forward processing.
#  ====================================================


def current_model(**kwds):
    def decorate(f):
        for k in kwds:
            if k == 'net':
                setattr(f, k, kwds[k])
        return f
    return decorate


@current_model(net="Model")
class Net(object):
    def __init__(self, model = None):
        if model is not None:
            self.model = model
        else:
            model_classname = getattr(Net, 'net', 0)
            model_class = getattr(model, model_classname)
            self.model = model_class.__new__(model_class)
        assert self.model is not None, "Please either create a model or use annotation @current_model"

    def inference(self, pre_proccessed_data):
        return self.model.inference(pre_proccessed_data)

    @staticmethod
    def pre_process(input_data):
        return input_data

    @staticmethod
    def post_process(result):
        return result

    def process(self, input_data):
        pre_processed_data = Net.pre_process(input_data)
        result = self.inference(pre_processed_data)
        return Net.post_process(result)


if __name__ == '__main__':
    pass
