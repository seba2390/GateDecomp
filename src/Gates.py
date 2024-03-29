import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse import kron as tensor_product
from scipy.sparse.linalg import expm as matrix_exponential

import warnings


class QuantumGate:
    def __init__(self, name: str, scaling: float = 1.0):
        self.name = name
        self.size = len(name)
        self._scaling_ = scaling
        self._matrix_representation_ = self._set_matrix_()

    def _set_matrix_(self):
        pass

    def get_size(self):
        return self.size

    def get_name(self):
        return self.name

    def matrix(self, as_dense: bool = True):
        if as_dense:
            return np.array(self._scaling_ * self._matrix_representation_.todense())
        return self._scaling_ * self._matrix_representation_

    def time_evolution(self, time: float = 1.0, as_dense: bool = True):
        if as_dense:
            return matrix_exponential(A=-1j * time * self.matrix())
        else:
            return csc_matrix(matrix_exponential(A=-1j * time * self.matrix(as_dense=False)))


class PauliGate(QuantumGate):
    _X_ = coo_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex64))
    _Y_ = coo_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex64))
    _Z_ = coo_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex64))
    _I_ = coo_matrix(np.array([[1, 0], [0, 1]], dtype=np.complex64))
    _allowed_gates_ = {'x': _X_, 'X': _X_,
                       'y': _Y_, 'Y': _Y_,
                       'z': _Z_, 'Z': _Z_,
                       'i': _I_, 'I': _I_}

    def __init__(self, name: str, scaling: float = 1.0):

        if len(name) == 1:
            if name not in list(self._allowed_gates_.keys()):
                raise ValueError(f'Unknown gate name provided, '
                                 f'should be one: {list(self._allowed_gates_.keys())}')
        elif len(name) > 1:
            for symbol in name:
                if symbol not in list(self._allowed_gates_.keys()):
                    raise ValueError(f'Unknown gate name provided, '
                                     f'should be one: {list(self._allowed_gates_.keys())}')
        else:
            raise ValueError(f'name does not have positive length.')
        super().__init__(name=name, scaling=scaling)

    def _set_matrix_(self):
        if len(self.name) == 1:
            return self._allowed_gates_[self.name]
        else:
            resulting_tensor_product = self._allowed_gates_[self.name[0]]
            for remaining_symbol in range(1, len(self.name)):
                resulting_tensor_product = tensor_product(resulting_tensor_product,
                                                          self._allowed_gates_[self.name[remaining_symbol]],
                                                          format=resulting_tensor_product.format)
            return resulting_tensor_product.asformat('csc')


class PauliExpression:
    @staticmethod
    def add(matrix1: csc_matrix, matrix2: csc_matrix) -> csc_matrix:
        return matrix1 + matrix2

    @staticmethod
    def subtract(matrix1: csc_matrix, matrix2: csc_matrix) -> csc_matrix:
        return matrix1 - matrix2

    _arithmetic_operators_ = {'+': add,
                              '-': subtract}
    _known_gates_ = list(PauliGate._allowed_gates_.keys())
    _allowed_tokens = list(_arithmetic_operators_.keys()) + _known_gates_

    def __init__(self, expression: str):
        for token in expression:
            if token not in self._allowed_tokens:
                raise ValueError(f'Unrecognized token in expression, should be one of: {self._allowed_tokens}')
        if expression[0] in self._arithmetic_operators_.keys() or expression[-1] in self._arithmetic_operators_.keys():
            raise ValueError(
                f'expression should neither begin or end with any of: {list(self._arithmetic_operators_.keys())}')

        self.expression = expression
        self._terms_ = {}
        self._operations_ = []
        self._set_terms_and_operations_()

    def _set_terms_and_operations_(self):

        term_counter = 0
        empty = ''
        for token in self.expression:
            if token in self._known_gates_:
                empty += token
            else:
                self._operations_.append(token)
                self._terms_[term_counter] = empty
                term_counter += 1
                empty = ''
        self._terms_[term_counter] = empty

        term_size = len(self._terms_[0])
        for term_idx in list(self._terms_.keys()):
            if len(self._terms_[term_idx]) != term_size:
                raise ValueError(f'Expression should only contain terms of equal length.')

    def matrix(self, as_dense: bool = True):
        current_term_matrix = PauliGate(name=self._terms_[0]).matrix(as_dense=False)
        for remaining in range(len(list(self._terms_.keys())[1:])):
            operation_idx, term_idx = remaining, remaining + 1
            operation = self._operations_[operation_idx]
            matrix = PauliGate(name=self._terms_[term_idx]).matrix(as_dense=False)
            current_term_matrix = self._arithmetic_operators_[operation](current_term_matrix, matrix)
        if as_dense:
            return np.array(current_term_matrix.todense())
        else:
            return current_term_matrix

    def time_evolution(self, time: float = 1.0, as_dense: bool = True):
        if as_dense:
            return matrix_exponential(A=-1j * time * self.matrix())
        else:
            return csc_matrix(matrix_exponential(A=-1j * time * self.matrix(as_dense=False)))

    def terms(self):
        return self._terms_

    def operations(self):
        return self._operations_

    @staticmethod
    def are_commuting(term1: str, term2: str) -> bool:
        """See Chapter 4.3 of https://arxiv.org/abs/1907.13623"""
        N_not_commuting = 0
        for gate1, gate2 in zip(term1, term2):
            if gate1 != gate2:
                if gate1 != 'i' and gate1 != 'I' and gate2 != 'i' and gate2 != 'I':
                    N_not_commuting += 1
        return N_not_commuting % 2 == 0

    def commutator_map(self):
        _commutator_map_ = {}
        N = len(list(self._terms_.keys()))
        for term1_idx in range(N):
            for term2_idx in range(term1_idx + 1, N):
                _commutator_map_[(term1_idx, term2_idx)] = self.are_commuting(self._terms_[term1_idx],
                                                                              self._terms_[term2_idx])
        return _commutator_map_

    def expectation(self, state_vector: np.ndarray):
        O = self.matrix()
        # Check if the matrix is Hermitian
        if not np.all(O == O.T) or not np.all(np.isreal(O)):
            warnings.warn('Note that matrix is not Hermitian', RuntimeWarning)

        # Check if the state vector is physical
        if not np.isclose(np.sum(np.abs(state_vector) ** 2), 1.0):
            warnings.warn('Note that state vector is not physical -> sum_i |c_i|^2 != 1', RuntimeWarning)

        return float(np.real_if_close(state_vector.T.conj() @ (O @ state_vector)))
