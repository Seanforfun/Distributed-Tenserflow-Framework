#  ====================================================
#   Filename: distribute_model.py
#   Function: This file is used to save the model of the gmean CNN
#   net.
#  ====================================================
import abc

import distribute_log as logger


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def inference(self, input_data):
        """
        The method must be implemented to generate the CNN model.
        This method implements the forward propagation.
        :param input_data: Raw data to train.
        :return: The temporary result processed by the model
        """
        pass


if __name__ == '__main__':
    pass
