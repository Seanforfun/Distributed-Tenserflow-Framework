#  ====================================================
#   Filename: distribute_input.py
#   Author: Botao Xiao
#   Function: This file contains the input module of the distributed
#   system.
#  ====================================================
import abc


class Input(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def load_data(data_dir, batch_size, gpu_num):
        """
        Users need to implement this method to get input and groundtruth
        data_dir: Please to load data, can be a string or tuple.
        :return: raw_data
        :return: ground_truth
        """
        pass
