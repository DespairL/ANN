import numpy as np
import struct
import os

"""
Cite from : http://yann.lecun.com/exdb/mnist/

TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
"""

TEST = 1
TRAIN = 0

"""
    对./MNIST_data目录下的文件进行针对性解码。
    Parameter :
        @id : 文件结构类型,对应于文件名中的idx
        @train_or_test : 标识训练集处理还是测试集处理
        @buffer : 文件流
    Return    :
        ret : 解析到的数据
"""
def decode(id, train_or_test, buffer):
    offset = 0
    if id == 3:
        format = '>iiii'
        _, image_number, image_row, image_col = struct.unpack_from(format, buffer, offset)
        # print(image_number, image_row, image_col)
        offset += struct.calcsize(format)
        ret = np.empty((image_number, image_row * image_col))
        loop_format = '>' + str(image_col * image_row) + 'B'
        for i in range(image_number):
            ret[i] = struct.unpack_from(loop_format, buffer, offset)
            # print(ret[i].shape)
            offset += struct.calcsize(loop_format)
        return ret
    else :
        format = '>ii'
        _, image_number = struct.unpack_from(format, buffer, offset)
        # print(image_number)
        offset += struct.calcsize(format)
        ret = np.empty((image_number))
        loop_format = '>B'
        for i in range(image_number):
            ret[i] = struct.unpack_from(loop_format, buffer, offset)[0]
            offset += struct.calcsize(loop_format)
        return ret

def generate_object(file):
    words = file.split('-')
    # print(words)
    id = 3 if words[1] == 'images' else 1
    train_or_test = TEST if words[0] == 't10k' else TRAIN
    ret = ""
    ret += "test" if train_or_test == TEST else "train"
    ret += "_x" if id == 3 else "_y"
    return ret, id , train_or_test
        
def load_data():
    base_path = "./MNIST_data/"
    files = os.listdir(base_path)
    id = -1
    train_or_test = 2
    return_dict = {'train_x':0,'train_y':0,'test_x':0,'test_y':0}
    for file in files:
        buffer = open(base_path + file, 'rb').read()
        object, id, train_or_test = generate_object(file)
        return_dict[object] = decode(id, train_or_test, buffer)
    return return_dict['train_x'], return_dict['train_y'], return_dict['test_x'], return_dict['test_y']
    # print(files)

def test():

    from torchvision import datasets, transforms

    def rigth_MNIST():
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
        train_x = dataset1.data.numpy()
        train_y = dataset1.targets.numpy()
        test_x = dataset2.data.numpy()
        test_y = dataset2.targets.numpy()
        # print(f"Right TrainDataset Shape : {train_x.shape}{train_y.shape}")
        # print(f"Right TestDataset Shape : {test_x.shape}{test_y.shape}")
        return train_x, train_y, test_x, test_y
    
    return rigth_MNIST()

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    # print(type(train_x), type(train_y), type(test_x), type(test_y))
    # train_x_test, train_y_test, test_x_test, test_y_test = test()

    # assert train_x.all() == train_x_test.all()
    # assert train_y.all() == train_y_test.all()
    # assert test_x.all()  == test_x_test.all()
    # assert test_y.all()  == test_y_test.all()

    # print("load MNIST dataset ok!")
    
