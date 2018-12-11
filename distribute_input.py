#  ====================================================
#   Filename: distribute_input.py
#   Author: Botao Xiao
#   Function: This file contains the input module of the distributed
#   system.
#  ====================================================
import abc
import tensorflow as tf


class Input(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def load_train_batch(data_dir, batch_size, param):
        """
        Users need to implement this method to get input and ground truth.
        data_dir: Place to load data, can be either a string or a tuple contains multiple paths.
        :param data_dir: Path to load data, can be either a string or a tuple saving multiple paths.
        :param batch_size: size of data in one batch.
        :param param (Optional) for user extension
        :return: raw_data batch
        :return: ground_truth batch
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def load_eval_batch(data_dir, batch_size, param):
        """
        Abstract method of loading evaluation batch, user must implement this function and return
        raw data and ground truth from the data paths.
        :param data_dir: Path to load data, can be either a string or a tuple saving multiple paths.
        :param batch_size: size of data in one batch.
        :param param (Optional) for user extension
        :return: raw_data batch in list
        :return: ground_truth (Optional)in list, Ground truth batch.
        """
        pass

    @staticmethod
    def _generate_image_batch(example_list, min_queue_examples, batch_size, num_thread, shuffle=True):
        if shuffle:
            examples = tf.train.shuffle_batch(
                example_list,
                batch_size=batch_size,
                num_threads=num_thread,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            examples = tf.train.batch(
                example_list,
                batch_size=batch_size,
                num_threads=num_thread.NUMBER_PREPROCESS_THREADS,
                capacity=min_queue_examples + 3 * batch_size
            )
        return examples
