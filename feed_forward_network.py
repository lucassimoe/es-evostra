import numpy as np

try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle


class FeedForwardNetwork(object):
    def __init__(self, layer_sizes):
        self.weights = []
        self.layer_sizes = layer_sizes
        for index in range(len(layer_sizes)-1):
            self.weights.append(np.zeros(shape=(layer_sizes[index], layer_sizes[index+1])))

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        for i, layer in enumerate(self.weights):
            out = np.dot(out, layer)
            out = np.arctan(out)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        weight_index = 0
        self.weights = []
        for index in range(len(self.layer_sizes)-1):
            layer = np.zeros(shape=(self.layer_sizes[index], self.layer_sizes[index+1]))            
            for i in range(self.layer_sizes[index]):
                for j in range(self.layer_sizes[index+1]):
                    layer[i][j] = weights[weight_index]
                    weight_index += 1
            self.weights.append(layer)

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
            
    def update_model_weights(self, new_weights):
        """
        Updates the network with new weights after they have been stored in one
        flat parameter vector
        """
        accum = 0
        for layer in model.layers:
            current_layer_weights_list = layer.get_weights()
            new_layer_weights_list = []
            for layer_weights in current_layer_weights_list:
                layer_total = np.prod(layer_weights.shape)
                new_layer_weights_list.append(
                    new_weights[accum:accum + layer_total].
                reshape(layer_weights.shape))
            accum += layer_total
        layer.set_weights(new_layer_weights_list)