class NetWork(object):
    def __init__(self):
        self.layer_list = []
        self.total_layer_num = 0

    def append(self, layer):
        self.layer_list.append(layer)
        self.total_layer_num += 1

    def forward(self, x):
        ret = x
        for index in range(self.total_layer_num):
            ret = self.layer_list[index].forward(ret)
        return ret

    def backward(self, grad):
        ret_grad = grad
        for index in range(self.total_layer_num-1, -1, -1):
            ret_grad = self.layer_list[index].backward(ret_grad)


    def update(self, parameter):
        for index in range(self.total_layer_num):
            if self.layer_list[index].can_update:
                self.layer_list[index].update(parameter)
    
    def get_weight_list(self):
        weight_list = []
        for index in range(self.total_layer_num):
            if self.layer_list[index].can_update:
                weight_list.append(self.layer_list[index].get_weight)
        return weight_list
    
    def get_final_weight(self):
        return self.layer_list[self.total_layer_num-1].get_weight()
     
    def get_final_grad_w(self):
        return self.layer_list[self.total_layer_num-1].get_grad_w()
    
    def set_final_grad_w(self, grad_w):
        self.layer_list[self.total_layer_num-1].grad_w = grad_w
