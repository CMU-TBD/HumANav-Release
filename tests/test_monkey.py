import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    tf.enable_eager_execution()
import threading, socket, time
from simulators.controller import Controller

"""BEGIN socket utils"""

def socket_send(message, port, host=None):
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Define host
    if(host is None):
        host = socket.gethostname()
    # Connect the socket to the port where the server is listening
    server_address = ((host, port))
    client_socket.connect(server_address)
    # Send data
    client_socket.sendall(bytes(str(message), "utf-8"))
    # Close communication channel
    client_socket.close()

def socket_listen(recieved, port, host=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Define host
    if(host is None):
        host = socket.gethostname()
    s.bind((host, port))
    s.listen(10)
    running = True # initialize listener
    while(running):
        connection, client = s.accept()
        while(True): # constantly taking in information until breaks
            # TODO: allow for buffered data, thus no limit
            data = connection.recv(128)
            # quickly close connection to open up for the next input
            connection.close()
            # NOTE: data is in the form (running, time, lin_command, ang_command)
            # TODO: use ast.literal_eval instead of eval to
            data = eval(data)
            recieved.append(data)
            # np_data = np.array([data[2], data[3]], dtype=np.float32)
            # # NOTE: commands can also be a dictionary indexed by time
            # self.commands.append(np_data)
            if(data[0] is False):
                running = False
                break
    s.close()

def establish_client_connection(port, host=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    h = None
    # Define host
    if(host is None):
        h = socket.gethostname()
    else:
        h = host
    server_address = ((h, port))
    s.connect(server_address)
    return s

def establish_server_connection(port, host=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    h = None
    # Define host
    if(host is None):
        h = socket.gethostname()
    else:
        h = host
    s.bind((h, port))
    s.listen(1)
    connection, client = s.accept()
    return connection, client

"""END socket utils"""

def test_monkey():
    s = establish_client_connection(6000)
    print("Controller->Simulator connection established")
    running = []
    power = threading.Thread(target=socket_listen, args=(running))
    # Create random controller
    # C = Controller()
    # C.update()
    while(False in running):
        time.sleep(0.01)
    s.close()

if __name__ == '__main__':
    test_monkey()