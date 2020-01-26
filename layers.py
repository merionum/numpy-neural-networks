import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(preds):
    dim = preds.ndim
    if dim == 1:
        c = np.max(preds)
        return np.exp(preds - c) / np.sum(np.exp(preds - c))
    sm = []
    for p in preds:
        c = np.max(p)
        sm.append(np.exp(p - c) / np.sum(np.exp(p - c)))
    return np.stack(sm, axis=0)


def cross_entropy_loss(predictions, targets):
    if predictions.ndim == 1:
        return -np.log(predictions[targets])
    N = predictions.shape[0]
    ce = -np.sum(np.log(predictions[np.arange(N), targets.flatten()]))  / N
    return ce


def softmax_with_cross_entropy(predictions, target_index):
    ndim = predictions.ndim
    sm = softmax(predictions)
    
    loss = cross_entropy_loss(sm, target_index)
    dprediction = np.zeros_like(sm)
    if ndim == 1:
        dprediction[target_index] = 1
    else:
        dprediction[np.arange(dprediction.shape[0]),
                    target_index.flatten()] = 1
    dprediction = sm - dprediction
    return loss, dprediction

def add_padding(X, pad):
    return np.pad(X, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant')

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.hidden = np.maximum(0, X)
        return self.hidden

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_out[self.hidden <= 0] = 0
        return d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.1*np.random.randn(n_input, n_output))
        self.B = Param(0.1*np.random.randn(1, n_output).flatten())
        self.X = None

    def forward(self, X):
        self.X = X
        self.f = ReLULayer() 
        result = self.f.forward(np.dot(self.X, self.W.value) + self.B.value)
        return result
    
    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        d_result = np.dot(self.f.backward(d_out), self.W.value.T)
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.stride = 1
        self.padding = padding


    def forward(self, X):
        self.X = X

        batch_size, height, width, channels = X.shape

        out_height = int((height - self.filter_size) / self.stride) + 1
        out_width = int((width - self.filter_size) / self.stride) + 1
        
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        self.f = ReLULayer()
        
        self.filters = np.reshape(self.W.value, (self.filter_size**2*channels, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                y_start, y_end, x_start, x_end = y, y+self.filter_size, x, x+self.filter_size

                receptive_area = X[:, y_start:y_end, x_start:x_end, :]
                receptive_area = np.reshape(receptive_area, (batch_size, self.filter_size**2*channels))
                
                #11, 12 dot  12, 2 => 11,2
                result = self.f.forward(receptive_area.dot(self.filters) + self.B.value)

                out[:, y, x, :] = result
        # ~ print(receptive_area.shape)
        # ~ print(self.filters.shape)
        # ~ print(out[:, y, x, :].shape)
        
        #padding
        return np.pad(out,mode='constant',pad_width=((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_result = np.zeros((batch_size, height, width, channels))

        # ~ print("np.dot(self.f.backward(d_out[:, y, x, :]), self.filters.T)")
        # ~ print(np.dot(self.f.backward(d_out[:, y, x, :]), self.filters.T).shape)
        # ~ print("self.f.backward(d_out[:, y, x, :]).shape")
        # ~ print(self.f.backward(d_out[:, y, x, :]).shape)
        # ~ print("self.filters.T.shape")
        # ~ print(self.filters.T.shape)
        # ~ print("d_result[:, y_start:y_end, x_start:x_end, :].shape")
        # ~ print(d_result[:, y_start:y_end, x_start:x_end, :].shape)

        for y in range(out_height):
            for x in range(out_width):
                y_start, y_end, x_start, x_end = y, y+self.filter_size, x, x+self.filter_size
                
                d_wrt_input = np.dot(self.f.backward(d_out[:, y, x, :]), self.filters.T)
                # ~ print("d_wrt_input", d_wrt_input.shape)
                # ~ print("")
                
                # ~ print("d_result[:, y_start:y_end, x_start:x_end, :]")
                # ~ print(d_result[:, y_start:y_end, x_start:x_end, :].shape)
                # ~ print("")
                d_wrt_input = np.reshape(d_wrt_input, d_result[:, y_start:y_end, x_start:x_end, :].shape)

                d_result[:, y_start:y_end, x_start:x_end, :] += d_wrt_input
                
                # ~ print("self.X[:, y_start:y_end, x_start:x_end, :]")
                # ~ print(self.X[:, y_start:y_end, x_start:x_end, :].shape)
                # ~ print("")
                
                # ~ print("d_out.shape")
                # ~ print(d_out.shape)
                # ~ print("")
                
                tmp = np.dot(self.X[:, y_start:y_end, x_start:x_end, :].T, d_out[:, y, x, :])
                # ~ print("tmp.shape")
                # ~ print(tmp.shape)
                # ~ print("")
                tmp=tmp.swapaxes(0,1)
                tmp=tmp.swapaxes(1,2)
                tmp=tmp.swapaxes(0,1)
                
                self.W.grad += tmp

                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)
        
        return d_result

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = self.X.shape

        self.pool_height = int((height - self.pool_size) / self.stride) + 1
        self.pool_width = int((width - self.pool_size) / self.stride) + 1

        maxpool = np.zeros((batch_size, self.pool_height, self.pool_width, channels))

        for i in range(batch_size):
            sample = self.X[i]

            for y in range(self.pool_height):
                for x in range(self.pool_width):
                    x_start, x_end, y_start, y_end = x, x+self.pool_size, y, y+self.pool_size
                    
                    for c in range(channels):
                        channel = sample[x_start:x_end, y_start:y_end, c]
                        max_in_channel = channel.max()
                        maxpool[i, y, x, c] = max_in_channel

        return maxpool

    def backward(self, d_out):
        
        batch_size, height, width, channels = self.X.shape

        dX = np.zeros((batch_size, height, width, channels))

        for i in range(batch_size):
            sample = self.X[i]

            for y in range(self.pool_height):
                for x in range(self.pool_width):
                    x_start, x_end, y_start, y_end = x, x+self.pool_size, y, y+self.pool_size
                    for c in range(channels):
                        pool = sample[x_start:x_end, y_start:y_end, c]
                        mask = (pool == np.max(pool))
                        dX[i, x_start:x_end, y_start:y_end, c] += d_out[i, y, x, c] * mask
        return dX
        
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = batch_size, height, width, channels = X.shape
        X = np.reshape(X, (batch_size, height*width*channels))
        return X

    def backward(self, d_out):
        return np.reshape(d_out, self.X_shape)

    def params(self):
        return {}
