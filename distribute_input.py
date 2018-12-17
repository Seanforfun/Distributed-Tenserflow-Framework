#  ====================================================
#   Filename: distribute_input.py
#   Author: Botao Xiao
#   Function: This file contains the input module of the distributed
#   system.
#  ====================================================
import os
import abc
import multiprocessing
from enum import Enum
import queue

import tensorflow as tf

import distribute_flags as flags
import distribute_constants as constants


class InputOptions(Enum):
    TF_RECORD = 0
    PLACEHOLDER = 1
    DATAPATHLOADER = 2


class Dataloader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_train_batch(self, name_queue=None, *args, **kwargs):
        """
        Users need to implement this method to get input and ground truth.
        data_dir: Place to load data, can be either a string or a tuple contains multiple paths.
        :param name_queue: (Optional) Load data from name queue.
        tf-record implementation: We create a queue using string_input_producer.
        placeholder implementation:
        :param args: (Optional) Additional parameters for training.
        :param kwargs: (Optional) Additional dict for training.
        :return: raw_data batch
        :return: ground_truth batch
        """
        pass

    @abc.abstractmethod
    def load_eval_batch(self, *args, **kwargs):
        """
        Abstract method of loading evaluation batch, user must implement this function and return
        raw data and ground truth from the data paths.
        :param data_dir: Path to load data, can be either a string or a tuple saving multiple paths.
        :param args: (Optional) Additional parameters for evaluation.
        :param kwargs: (Optional) Additional dict for evaluation.
        :return: raw_data batch in list
        :return: ground_truth (Optional)in list, Ground truth batch.
        """
        pass

    def _generate_image_batch(self, example_list, min_queue_examples, num_thread, shuffle=True):
        if shuffle:
            examples = tf.train.shuffle_batch(
                example_list,
                batch_size=self.batch_size,
                num_threads=num_thread,
                capacity=min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=0)
        else:
            examples = tf.train.batch(
                example_list,
                batch_size=self.batch_size,
                num_threads=num_thread.NUMBER_PREPROCESS_THREADS,
                capacity=min_queue_examples + 3 * self.batch_size
            )
        return examples


class TFRecordDataLoader(Dataloader, metaclass=abc.ABCMeta):
    type = "TFRecordDataLoader"

    def load_train_batch(self, name_queue=None, *args, **kwargs):
        if name_queue is None:
            raise RuntimeError("Cannot find get the queue from tf-record.")
        raw_data, ground_truth = self.load_batch_from_tfrecord(name_queue)
        return raw_data, ground_truth

    def load_eval_batch(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __decode_raw_data(self, raw_features, height, width, *args, **kwargs):
        """
        :param raw_features: raw examples retrieved from tf-record file
        :return: sample list [raw data, grounp truth]
        """
        pass

    def load_batch_from_tfrecord(self, filename_queue, *args, **kwargs):
        height = flags.FLAGS.input_image_height
        width = flags.FLAGS.input_image_width
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        raw_features = tf.parse_single_example(
            serialized_example,
            self.features)
        example_list = self.__decode_raw_data(raw_features, height, width, args, kwargs)
        min_queue_examples = int(self.sample_number * constants.MIN_FRACTION_OF_EXAMPLE_IN_QUEUE)
        batch_data = self._generate_image_batch(example_list,
                                                min_queue_examples,
                                                multiprocessing.cpu_count() * 2,
                                                shuffle=True)
        return batch_data


class DataPathDataLoader(Dataloader, metaclass=abc.ABCMeta):
    type = "DataPathDataLoader"

    def load_train_batch(self, name_queue=None, *args, **kwargs):
        assert name_queue is not None, "name queue cannot be None for DataPathDataLoader!"
        reader = self.__create_reader()
        key, value = reader.read()
        example_list = self.__parse_raw_data(value)
        min_queue_examples = int(self.sample_number * constants.MIN_FRACTION_OF_EXAMPLE_IN_QUEUE)
        return self._generate_image_batch(example_list, min_queue_examples,
                                          multiprocessing.cpu_count() * 2,
                                          shuffle=True)

    def load_eval_batch(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __parse_raw_data(self, raw_reader_data):
        """
        Parse the raw data from reader and decode them to tensors which can be used directly
        in the network.
        :param raw_reader_data: raw_data is directly from reader.
        :return: example_list
        """
        pass

    @abc.abstractmethod
    def __create_reader(self):
        """
        :return: Create a reader for parsing data listed in the queue.
        """
        pass

    @abc.abstractmethod
    def create_name_queue(self, data_dir):
        """
        Create a name queue for reader to read data from there.
        :return: A name queue saving all full path to load the data.
        """
        pass


class PlaceholderDataLoader(Dataloader, metaclass=abc.ABCMeta):
    type = "PlaceholderDataLoader"

    def load_train_batch(self, name_queue=None, *args, **kwargs):
        return self.__create_placeholder(args, kwargs)

    def load_eval_batch(self, *args, **kwargs):
        pass

    def load_queue_for_placeholder(self, *args, **kwargs):
        batch_queue = queue.Queue()
        self.__put_names_dict_into_queue(batch_queue, args, kwargs)
        return batch_queue

    @abc.abstractmethod
    def __create_placeholder(self, *args, **kwargs):
        """
        User must implement this method and return the placeholder
        :param args: (Optional) User's parameter. Save height, width etc.
        :param kwargs: (Optional) User's dict. Save height, width etc.
        :return: return placeholders
        """
        pass

    @abc.abstractmethod
    def __put_names_dict_into_queue(self, queue, *args, **kwargs):
        """
        Users must implement this method so that all datas path are arranged as dictionary
        and save into the queue.
        :param queue: queue to insert the samples.
        :param args: (Optional) For user to pass extension parameters.
        :param kwargs: (Optional) For user to pass extension dict.
        :return:
        """
        pass

    @abc.abstractmethod
    def decode_data_from_path_name(self, paths, *args, **kwargs):
        """
        We get the sample name queue at the very begining, now we get the data
        from the path and return them so program can generate a queue.
        :param paths: paths of the data to read from.
        :param args: (Optional) For user to pass extension parameters.
        :param kwargs: (Optional) For user to pass extension dict.
        :return: data Dict, two items {'raw_data': [], 'ground_truth': []}
        """
        pass

    def load_placeholder_data(self, sample_path_queue, *args, **kwargs):
        raw_batch = []
        ground_truth_batch = []
        for i in range(self.batch_size):
            paths = sample_path_queue.get()
            data = self.decode_data_from_path_name(paths, args, kwargs)
            raw_batch.append(data['raw_data'])
            ground_truth_batch.append(data['ground_truth'])
            sample_path_queue.put(paths)
        return raw_batch, ground_truth_batch

