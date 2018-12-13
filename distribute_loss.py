#  ====================================================
#   Filename: distribute_train.py
#   Author: Botao Xiao
#   Function: The training file is used to save the training process
#  ====================================================

import abc


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def loss_fn(self, predict, ground_truth):
        """
        :param predict: The predict result from the net structure.
        :param ground_truth: ground_truth value to compare with.
        :return: loss value
        """
        pass
