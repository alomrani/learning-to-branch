
from sklearn.neural_network import MLPClassifier
import numpy as np



class MLPClassifier1(MLPClassifier):
    def __init__(self, init_params=None, hidden_layer_sizes=(100,), activation="relu", *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)
        self.init_params = init_params
    def _init_coef(self, fan_in, fan_out, dtype):
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.0
        if self.activation == "logistic":
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))
        params_layer_idx = {(72, 100): 0, (100, 1): 1}
        # Generate weights and bias:
        if self.init_params is not None:
            coef_init = np.asarray(self.init_params[0][params_layer_idx[(fan_in, fan_out)]])
            intercept_init = np.asarray(self.init_params[1][params_layer_idx[(fan_in, fan_out)]])
        else:
            coef_init = self._random_state.uniform(
                -init_bound, init_bound, (fan_in, fan_out)
            )

            intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init