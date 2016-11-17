import numpy as np

class _Utilities():
    def _target_array_validation(self, y, allowed=None):
        if not np.issubdtype(y[0], int):
            raise AttributeError('y must be an integer array.\nFound %s'
                                 % y.dtype)
        found_labels = np.unique(y)
        if (found_labels < 0).any():
            raise AttributeError('y array must not contain negative labels.'
                                 '\nFound %s' % found_labels)
        if allowed is not None:
            found_labels = tuple(found_labels)
            if found_labels not in allowed:
                raise AttributeError('Labels not in %s.\nFound %s'
                                     % (allowed, found_labels))
                
    def _sse_cost(self, y, target):
            errors = (target - y)
            return (errors**2).sum() / 2.0
