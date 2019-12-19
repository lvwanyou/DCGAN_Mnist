

"""
Helper for data pre-processing!
"""
import math
import re
import os
import sys
import struct
import importlib
import numpy as np
import tensorflow as tf

hex_pattern = re.compile(r'^[a-fA-F0-9]+$')
importlib.reload(sys)  ########################################################################
# sys.setdefaultencoding( "ISO-8859-1" )   #################################################################################
# sys.setdefaultencoding( "utf-8" )


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name)



def files_number(dirPath, fileName=None):
    """
    get file number in dir
    :param  dirPath; filename
    :return: file number.
    """
    if fileName is None:
        return len(os.listdir(dirPath))
    else:
        count = 0
        for item in os.listdir(dirPath):
            if item.__contains__(fileName):
                count += 1
        return count


def convert_to_one_hot(Y, C):
    """
    Convert  hex to one hot
    :param string_num: Y - target  ;  C : one hot dimension number
    :return: one hot vector
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def dec2hex(string_num):
    """
    Convert int number to a hex string.
    :param string_num:
    :return:
    """
    # if string_num.isdigit():
    hex_str = hex(string_num)
    hex_str = hex_str.replace('0x', '')
    if len(hex_str) < 2:
        hex_str = '0' + hex_str
    return hex_str


def hex2dec(string_num):
    """
    Convert a hex string to a int number
    :param string_num:
    :return: -1 if the string_num given is illegal
    """
    if hex_pattern.match(string_num):
        return int(string_num.upper(), 16)
    else:
        return -1


def hex2dec_on_list(lst):
    """
    Conduct the hex2dec operation on a list.
    :param lst:
    :return:
    """
    data = []
    for i, val in enumerate(lst):
        data.append(hex2dec(val))
    return data

def transfer(path):
    """
    Transfer all the str in the file to separated version.(Separated by space per two chars)
    :param path:
    :return:
    """
    file_name = 'separated_modbus.txt'
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, file_name)
    new_content = ''
    with open(path, 'r') as f:
        try:
            text = f.readlines()
            for index, val in enumerate(text):
                new_str = divide_str(val)
                new_content += new_str
        except IOError:
            print('can not read the file!')


def divide_str(string):
    """
    Use space to separate the str per two chars.
    :param string:
    :return:
    """
    new_str = ''
    for i in range(0, int(len(string)), 2):
        new_str += string[i:i + 2] + ' '
    return new_str


def str2list(string):
    """
    Break the str into sub str of length 2 and form a list.
    :param string:
    :return:
    """
    return [string[i:i + 2] for i in range(0, len(string), 2)]

def convert_list_to_unicode_str(data):
    """
    Convert string to unicode str.
    :param data:
    :return:
    """
    string = ''
    for i, val in enumerate(data):
        # string = string + unicode(unichr(int(val)))
        string = string + str(int(val))
    return string

def make_request_string(string):
    """
    Wrap the string to a request string.
    :param string:
    :return:
    """
    message = str2list(string)
    modbus = hex2dec_on_list(message)
    return convert_list_to_unicode_str(modbus)


def dataSwitch(data):
    str1 = ''
    str2 = b''#############################################################################################################
    while data:
        str1 = data[0:2]
        s = int(str1,16)
        str2 += struct.pack('B',s)
        data = data[2:]
    return str2


class DataLoad(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.vector_array = []
        self.num_batch = 0
        self.vector_matrix = []
        # self.pointer = 0

    def create_batches(self, file_name):
        self.vector_array = []  #每一行的数据读到这个 list中，最后通过np.array把他转变为矩阵

        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                int_line = [int(x) for x in line]
                print("int_line........")
                print(int_line)
                self.vector_array.append(int_line)

        # 此处进行分块处理
        self.num_batch = int(len(self.vector_array) / self.batch_size)
        self.vector_array = self.vector_array[:self.num_batch * self.batch_size]#这一行好像没什么用，有用，应该是凑够整数个batch，不够的就丢掉了
        # self.vector_matrix = np.split(np.array(self.vector_array), self.num_batch, 0)
        # self.pointer = 0

    def random_mini_batches(self, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        # m = X.shape[1]  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(len(self.vector_array)))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(
            len(self.vector_array) / self.batch_size)  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    # def next_batch(self):
    #     ret = self.vector_matrix[self.pointer]
    #     self.pointer = (self.pointer + 1) % self.num_batch
    #     return ret
    #
    # def reset_pointer(self):
    #     self.pointer = 0