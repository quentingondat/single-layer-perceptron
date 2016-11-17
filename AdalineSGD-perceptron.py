import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from _tools import _Utilities

class AdalineSGD(_Utilities):
    
    def __init__(self, eta=0.01, epochs=100):
        self.eta = eta
        self.epochs = epochs
    
    def _fit(self, X, y, reinitialize_weights=True):
        self._target_array_validation(y, allowed={(0, 1)})
        y_q = np.where(y == 0, -1., 1.)
        
        if reinitialize_weights:
            self.w_ = np.zeros(1 + X.shape[1])

        self.costs_ = []
        
        for _ in tqdm(range(self.epochs)):
            for xi, t in zip(X, y_q):
                y_v = self._activation(xi)
                error = t - y_v
                self.w_[1:] += self.eta * xi.dot(error)
                self.w_[0] += self.eta * error
                
            self.costs_.append(self._sse_cost(self._activation(X), y_q))
            
        return self

    def _activation(self, X):
        return (np.dot(X, self.w_[1:]) + self.w_[0])
        
    def _predict(self, X):
        return np.where(self._activation(X) < 0.0, 0, 1)
        
    def _plot_error(self):
        plt.plot(range(1, len(ppn.costs_)+1), ppn.costs_)
        plt.xlabel('Epochs')
        plt.ylabel('SSE')
        plt.show()
