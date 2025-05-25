'''
from mpmath import zetazero, zeta
import numpy as np

# Tracer la partie réelle et imaginaire de la fonction zêta de Riemann sur la droite critique
import matplotlib.pyplot as plt

t = np.linspace(0.1, 40, 1000)
s = 0.5 + 1j * t
zeta_vals = np.array([zeta(val) for val in s])

plt.figure(figsize=(12, 6))
plt.plot(t, np.real(zeta_vals), label="Re(ζ(0.5+it))")
plt.plot(t, np.imag(zeta_vals), label="Im(ζ(0.5+it))")
plt.xlabel("t")
plt.ylabel("Valeur")
plt.title("Fonction zêta de Riemann sur la droite critique (s=0.5+it)")
plt.legend()
plt.grid()
plt.show()

def plot_decision_boundary():
    plt.figure(figsize=(10, 10))
    plt.contourf(W11, W22, classification, alpha=0.5, cmap='summer')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
    plt.scatter(Params1, Params2, c='red', s=10)
    plt.scatter(W[0], W[1], c='blue', s=100)
    plt.title('Decision Boundary ')
    marker = 'open', 'blue'
    plt.scatter(W[0], W[1], c=marker[1], s=100, edgecolor=marker[0])
    plt.show()
   
   
    kwargs  = {
        'contourf': {
            'X': W11,
            'Y': W22,
            'Z': classification,
            'alpha': 0.5,
            'cmap': 'summer'
        },
        'scatter': {
            'x': X[:, 0],
            'y': X[:, 1],
            'c': y,
            'cmap': 'summer'
        },
        'scatter_params': {
            'x': Params1,
            'y': Params2,
            'c': 'red',
            's': 10
        },
        'scatter_W': {
            'x': W[0],
            'y': W[1],
            'c': 'blue',
            's': 100
        },
        'title': "Decision Boundary",
        'scatter_marker': {
            'x': W[0],
            'y': W[1],
            'c': marker[1],
            's': 100,
            'edgecolor': marker[0]
        }
    }
    
    plt.figure(**kwargs)
    plt.contourf(**kwargs['contourf'])
    plt.scatter(**kwargs['scatter'])
    plt.scatter(**kwargs['scatter_params'])
    plt.scatter(**kwargs['scatter_W'])
    plt.title(kwargs['title'])
    plt.scatter(**kwargs['scatter_marker'])
    plt.show()
    
    
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def sigmoid_derivative(x):
            return x * (1 - x)
        def forward(X):
            self.hidden = sigmoid(np.dot(X, self.W1) + self.b1)
            self.output = sigmoid(np.dot(self.hidden, self.W2) + self.b2)
            return self.output
        def backward(X, y, learning_rate): 
            output_size = self.output.shape[1]
            output_error = y - self.output 
            output_delta = output_error * sigmoid_derivative(self.output)
            hidden_error = output_delta.dot(self.W2.T)
        def classify(X):
            return np.argmax(self.forward(X), axis=1)
        def train(X, y, epochs, learning_rate):
            for epoch in range(epochs):
                self.forward(X)
                self.backward(X, y, learning_rate)
                if epoch % 1000 == 0:
                    loss = np.mean(np.square(y - self.output))
                    print(f"Epoch {epoch}, Loss: {loss}")
                    self.W1 += X.T.dot(self.hidden) * learning_rate
                    self.b1 += np.sum(self.hidden, axis=0) * learning_rate
                elif epoch % 100 == 0:
                    loss = np.mean(np.square(y - self.output))
                    print(f"Epoch {epoch}, Loss: {loss}")
                    self.W2 += self.hidden.T.dot(output_delta) * learning_rate
                    self.b2 += np.sum(output_delta, axis=0) * learning_rate
                elif epoch % 10 == 0:
                    loss = np.mean(np.square(y - self.output))
                    print(f"Epoch {epoch}, Loss: {loss}")
                    self.W1 += X.T.dot(self.hidden) * learning_rate
                    self.b1 += np.sum(self.hidden, axis=0) * learning_rate
                else:
                    loss = np.mean(np.square(y - self.output))
                    print(f"Epoch {epoch}, Loss: {loss}")
                    self.W2 += self.hidden.T.dot(output_delta) * learning_rate
                    self.b2 += np.sum(output_delta, axis=0) * learning_rate
                def predict(self, X):
                    return np.argmax(self.forward(X), axis=1)
                def accuracy(self, X, y):
                    predictions = self.predict(X)
                    return np.mean(predictions == y)
                def plot_loss(self, loss):
                    plt.plot(loss)
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Loss over epochs')
                    plt.show()
                    def plot_decision_boundary(self, X, y):
                        plt.figure(figsize=(10, 10))
                        plt.contourf(W11, W22, classification, alpha=0.5, cmap='summer')
                        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
                        plt.scatter(Params1, Params2, c='red', s=10)
                        plt.scatter(W[0], W[1], c='blue', s=100)
                        plt.title('Decision Boundary ')
                        marker = 'open', 'blue'
                        plt.scatter(W[0], W[1], c=marker[1], s=100, edgecolor=marker[0])
                        plt.show()
            def plot_decision_boundary(self, X, y):
                plt.figure(figsize=(10, 10))
                plt.contourf(W11, W22, classification, alpha=0.5, cmap='summer')
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
                plt.scatter(Params1, Params2, c='red', s=10)
                plt.scatter(W[0], W[1], c='blue', s=100)
                plt.title('Decision Boundary ')
                marker = 'open', 'blue'
                plt.scatter(W[0], W[1], c=marker[1], s=100, edgecolor=marker[0])
                plt.show()
                
                def plot_decision_boundary(self, X, y):
                        plt.figure(figsize=(10, 10))
                        plt.contourf(W11, W22, classification, alpha=0.5, cmap='summer')
                        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
                        plt.scatter(Params1, Params2, c='red', s=10)
                        plt.scatter(W[0], W[1], c='blue', s=100)
                        plt.title('Decision Boundary ')
                        marker = 'open', 'blue'
                        plt.scatter(W[0], W[1], c=marker[1], s=100, edgecolor=marker[0])
                        plt.show()
        class NeuralNetwork:
            def __init__(self, input_size, hidden_size, output_size):
                self.W1 = np.random.randn(input_size, hidden_size)
                self.b1 = np.random.randn(hidden_size)
                self.W2 = np.random.randn(hidden_size, output_size)
                self.b2 = np.random.randn(output_size)
                
                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))
                def sigmoid_derivative(x):
                    return x * (1 - x)
                def forward(X):
                    self.hidden = sigmoid(np.dot(X, self.W1) + self.b1)
                    self.output = sigmoid(np.dot(self.hidden, self.W2) + self.b2)
                    return self.output
                def backward(X, y, learning_rate): 
                    output_size = self.output.shape[1]
                    output_error = y - self.output 
                    output_delta = output_error * sigmoid_derivative(self.output)
                    hidden_error = output_delta.dot(self.W2.T)
                    
                    hidden_delta = hidden_error * sigmoid_derivative(self.hidden)
                    self.W2 += self.hidden.T.dot(output_delta) * learning_rate
                    self.b2 += np.sum(output_delta, axis=0) * learning_rate
                    '''
                    
                