import numpy as np

""" 利用yield模拟dataloader """
def Dataloader(x, y, batch_size, shuffle=True):
    index_list = list(range(len(x)))
    if shuffle:
        np.random.shuffle(index_list)
    for index in range(0, len(x), batch_size):
        end_index = min(index + batch_size, len(x))
        """ (100,784) (100,) """
        yield x[index_list[index:end_index]], y[index_list[index:end_index]]

if __name__ == '__main__':
    from load_data import load_data
    train_x, train_y, test_x, test_y = load_data()
    dataloader = Dataloader(test_x, test_y, batch_size=100)
    for mini_x, mini_y in dataloader:
        print(mini_x.shape, mini_y.shape)
    dataloader2 = Dataloader(train_x, train_y, batch_size=100)
    for mini_x, mini_y in dataloader2:
        print(mini_x.shape, mini_y.shape)