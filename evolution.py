#!/usr/bin/env python3

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def init_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)

def main():
    init_tensorflow()

if __name__ == "__main__":
    main()
