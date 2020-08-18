import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    tf.enable_eager_execution()
import threading
import socket
import time
from simulators.joystick import Joystick


def test_joystick():
    J = Joystick()
    J.establish_robot_sender_connection()
    J.establish_robot_receiver_connection()
    # init control pipeline after recieved map from robot
    J.init_control_pipeline()
    J.update(random_commands=False)


if __name__ == '__main__':
    test_joystick()
