import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    tf.enable_eager_execution()
import threading, socket, time
from simulators.controller import Controller
from utils.utils import print_colors

def test_monkey():
    C = Controller(host=None, port=6000)
    C.establish_robot_connection()
    C.update()
    
if __name__ == '__main__':
    test_monkey()