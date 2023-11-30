import numpy as np


class State:
    # Pauli-z eigen basis.
    __allowed_states__ = {
        '0': np.array([[1], [0]], dtype=np.complex64),
        '1': np.array([[0], [1]], dtype=np.complex64)
    }

    # First element in list has eigenvalue +1, and second has eigenvalue -1.
    __basis_vectors__ = {
        'Z': [__allowed_states__['0'], __allowed_states__['1']],
        'X': [1. / np.sqrt(2) * np.array([[1], [1]], dtype=np.complex64),
              1. / np.sqrt(2) * np.array([[1], [-1]], dtype=np.complex64)],
        'Y': [1. / np.sqrt(2) * np.array([[1], [1j]], dtype=np.complex64),
              1. / np.sqrt(2) * np.array([[1], [-1j]], dtype=np.complex64)]
    }

    # Corresponding projection operators
    __projectors__ = {
        'Z': [__basis_vectors__['Z'][0] @ __basis_vectors__['Z'][0].T,
              __basis_vectors__['Z'][1] @ __basis_vectors__['Z'][1].T],
        'X': [__basis_vectors__['X'][0] @ __basis_vectors__['X'][0].T,
              __basis_vectors__['X'][1] @ __basis_vectors__['X'][1].T],
        'Y': [__basis_vectors__['Y'][0] @ __basis_vectors__['Y'][0].T,
              __basis_vectors__['Y'][1] @ __basis_vectors__['Y'][1].T]
    }

    def __init__(self, name: str, scaling: float = 1.0):
        if not isinstance(name, str):
            raise ValueError("State name must be a string.")

        self.name = name
        self.size = len(name)
        self._scaling_ = scaling
        self._state_vector_ = self._set_state_()

    def _set_state_(self):
        if self.size == 0:
            raise ValueError("State name cannot be empty.")

        state_vector = None
        for char in self.name:
            if char not in self.__allowed_states__:
                raise ValueError(f"Invalid state '{char}'. Allowed states are {list(self.__allowed_states__.keys())}.")

            char_state = self.__allowed_states__[char]
            if state_vector is None:
                state_vector = char_state
            else:
                state_vector = np.kron(state_vector, char_state)

        return self._scaling_ * state_vector

    def get_state_vector(self):
        return self._state_vector_

    def act_with(self, operator: np.ndarray):
        if not self._state_vector_.shape[0] == operator.shape[0] == operator.shape[1]:
            raise ValueError("Operator must be a square matrix of the same dimension as the state vector.")
        self._state_vector_ = np.dot(operator, self._state_vector_)

    def get_expectation(self, operator: np.ndarray) -> np.complex64:
        if not self._state_vector_.shape[0] == operator.shape[0] == operator.shape[1]:
            raise ValueError("Operator must be a square matrix of the same dimension as the state vector.")
        psi_f = np.dot(operator, self._state_vector_).flatten()
        return np.dot(self._state_vector_.flatten().conj(), psi_f)
