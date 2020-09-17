import h5py
import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.m = 6
        self.cost = list()
        self.parameters = dict()

    @staticmethod
    def _sigmoid(Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def _sigmoid_derivative(self, dA, Z):
        A = self._sigmoid(Z)
        return dA * A * (1 - A)

    @staticmethod
    def _relu(Z):
        A = np.maximum(0, Z)
        return A

    @staticmethod
    def _relu_derivative(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _forward_propagation(self, W, A, B, activation):
        Z = np.dot(W, A) + B
        if activation == "sigmoid":
            A = self._sigmoid(Z)
        elif activation == "relu":
            A = self._relu(Z)
        return A, Z

    def _backward_propagation(self, W, A_prev, dZ):
        m = self.m
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    # def _update_parameters(self, grads):
    #     self.parameters['W'+str(l+1)] -= learning_rate * \
    #         grads['dW'+str(l+1)]
    #     self.parameters['b'+str(l+1)] -= learning_rate * \
    #         grads['db'+str(l+1)]

    def _initialize_parameters(self, layer_dims):
        np.random.seed(1)
        params = dict()
        L = len(layer_dims)
        for l in range(1, L):
            params['W'+str(l)] = np.random.randn(layer_dims[l],
                                                 layer_dims[l-1]) * 0.01
            params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        return params

    def _compute_cost(self, Y, AL):
        m = self.m
        cost = (1/m) * (-np.dot(Y, np.log(AL).T) -
                        np.dot(1-Y, np.log(1-AL).T))
        return np.squeeze(cost)

    def fit(self, X, Y, layer_dims, iterations=1000, learning_rate=0.01, activation="relu"):
        self.m = X.shape[1]
        L = len(layer_dims)
        layer_dims.insert(0, X.shape[0])
        self.parameters = self._initialize_parameters(layer_dims)
        cache = {
            'A0': X
        }
        grads = {}

        for i in range(iterations):
            # Forward Propagation
            for l in range(1, L):
                cache['A'+str(l)], cache['Z'+str(l)] = self._forward_propagation(
                    self.parameters['W' + str(l)],
                    cache['A'+str(l-1)],
                    self.parameters['b'+str(l)],
                    activation='relu'
                )

            cache['A'+str(L)], cache['Z'+str(L)] = self._forward_propagation(
                self.parameters['W' + str(L)],
                cache['A'+str(L-1)],
                self.parameters['b'+str(L)],
                activation='sigmoid'
            )

            # Computing Cost
            cost = self._compute_cost(Y, cache['A' + str(L)])

            # BackPropagation
            AL = cache['A'+str(L)]
            grads['dA'+str(L)] = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
            grads['dZ'+str(L)] = self._sigmoid_derivative(
                grads['dA'+str(L)], cache['Z'+str(L)])

            grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db' + str(L)] = self._backward_propagation(
                self.parameters['W'+str(L)],
                cache['A'+str(L-1)],
                grads['dZ'+str(L)]
            )

            for l in reversed(range(1, L)):
                if activation == "sigmoid":
                    grads['dZ'+str(l)] = self._sigmoid_derivative(
                        grads['dA'+str(l)],
                        cache['Z'+str(l)]
                    )
                elif activation == "relu":
                    grads['dZ'+str(l)] = self._relu_derivative(
                        grads['dA'+str(l)],
                        cache['Z'+str(l)]
                    )

                grads['dA'+str(l-1)], grads['dW'+str(l)], grads['db'+str(l)] = self._backward_propagation(
                    self.parameters['W' + str(l)],
                    cache['A' + str(l-1)],
                    grads['dZ' + str(l)]
                )

            # Update Parameters
            for l in range(L):
                self.parameters['W'+str(l+1)] -= learning_rate * \
                    grads['dW'+str(l+1)]
                self.parameters['b'+str(l+1)] -= learning_rate * \
                    grads['db'+str(l+1)]

            # print Cost
            if i % 50 == 0:
                self.cost.append(cost)
                print(f"The Cost after {i} iteration is: {cost}")

    def predict(self, X, y):
        params = self.parameters
        L = len(params)//2
        cache = {
            'A0': X
        }
        m = self.m
        p = np.zeros((1, m))
        for l in range(1, L):
            cache['A'+str(l)], cache['Z'+str(l)] = self._forward_propagation(
                params['W' + str(l)],
                cache['A'+str(l-1)],
                params['b'+str(l)],
                activation='relu'
            )

        cache['A'+str(L)], cache['Z'+str(L)] = self._forward_propagation(
            params['W' + str(L)],
            cache['A'+str(L-1)],
            params['b'+str(L)],
            activation='sigmoid'
        )
        probas = cache['A'+str(L)]
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print("Accuracy: " + str(np.sum((p == y)/m)))
        return p


if __name__ == "__main__":
    np.random.seed(1)

    X = np.array([
        [1., 0., 0., 0., -10., 9.],
        [0., 1., 0., -6., -4., -1.]
    ])
    Y = np.array([1, 1, 1, 0, 0, 1])
    Y = Y.reshape(1, 6)

    nn = NeuralNetwork()
    print(nn.parameters)
    nn.fit(X, Y, layer_dims=[
           2, 1], iterations=50, learning_rate=0.01, activation="relu")
    print(nn.parameters)
    p = nn.predict(X, Y)
