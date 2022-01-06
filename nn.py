from collections import defaultdict
from nn_layer import *


def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return (-ones_for_answers + softmax) / logits.shape[0]


def logger(fnc):
    def wrapper(*args, **kwargs):
        logs = fnc(*args, **kwargs)
        res = defaultdict(list)
        for log in logs:
            for key in log.keys():
                print(key, log[key])
                res[key].append(log[key])
            print()
        return res
    return wrapper


class NN(object):
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        activations = []
        input = X

        for l in self.layers:
            activations.append(l.forward(input))
            input = activations[-1]

        assert len(activations) == len(self.layers)
        return activations

    def train_batch(self, X, y):
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations
        logits = layer_activations[-1]

        y_argmax = y.argmax(axis=1)
        loss = softmax_crossentropy_with_logits(logits, y_argmax)
        loss_grad = grad_softmax_crossentropy_with_logits(logits, y_argmax)

        for layer_index in range(len(self.layers))[::-1]:
            layer = self.layers[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)
        return np.mean(loss)

    @logger
    def train(self, X_train, y_train, n_epochs=25, batch_size=32):

        for epoch in range(n_epochs):
            loss = []
            for i in range(0, X_train.shape[0], batch_size):
                x_batch = np.array([x.flatten() for x in X_train[i:i + batch_size]])
                y_batch = np.array([y for y in y_train[i:i + batch_size]])
                loss.append(self.train_batch(x_batch, y_batch))

            epoch_log = {'epoch': epoch,
                                 'accuracy': np.mean(self.predict(X_train) == y_train.argmax(axis=-1)),
                                 'loss': np.mean(loss)}

            yield epoch_log

    def predict(self, X):
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)





