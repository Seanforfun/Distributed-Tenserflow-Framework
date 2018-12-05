#  ====================================================
#   Filename: distribute_train.py
#   Author: Botao Xiao
#   Function: The training file is used to save the training process
#  ====================================================

import tensorflow as tf

import distribute_log as logger


def train():
    logger.info("Start training process.")
    tf.reset_default_graph()
    with tf.Graph().as_default():
        pass


if __name__ == '__main__':
    pass
