import numpy as np


class Matrix:
    def __init__(self, data):
        matrix = np.asarray(data, dtype=float)

        if matrix.ndim != 2:
            raise ValueError(
                "Matrix phải là mảng 2 chiều."
            )

        self.data = matrix

    # ==========================================
    # BASIC PROPERTIES
    # ==========================================

    @property
    def rows(self):
        return self.data.shape[0]

    @property
    def columns(self):
        return self.data.shape[1]

    @property
    def shape(self):
        return self.data.shape

    @property
    def dimension(self):
        return self.data.shape

    def is_square(self):
        return self.rows == self.columns

    # ==========================================
    # MATRIX PROPERTIES
    # ==========================================

    def is_zero(self):
        return np.allclose(self.data, 0)

    def is_diagonal(self):
        if not self.is_square():
            return False

        return np.allclose(
            self.data,
            np.diag(np.diagonal(self.data))
        )

    def is_identity(self):
        if not self.is_square():
            return False

        return np.allclose(
            self.data,
            np.eye(self.rows)
        )

    def is_symmetric(self):
        """
        A symmetric iff A = A^T
        """

        if not self.is_square():
            return False

        return np.allclose(
            self.data,
            self.data.T
        )

    def is_singular(self):
        """
        Matrix singular iff det(A) = 0
        """

        if not self.is_square():
            raise ValueError(
                "Chỉ kiểm tra singular với matrix vuông."
            )

        return np.isclose(
            self.determinant(),
            0
        )

    def is_invertible(self):
        return not self.is_singular()

    # ==========================================
    # TRANSPOSE
    # ==========================================

    def transpose(self):
        return Matrix(self.data.T)

    # ==========================================
    # TRACE
    # ==========================================

    def trace(self):
        if not self.is_square():
            raise ValueError(
                "Trace chỉ tồn tại với matrix vuông."
            )

        return np.trace(self.data)

    # ==========================================
    # DETERMINANT
    # ==========================================

    def determinant(self):
        if not self.is_square():
            raise ValueError(
                "Determinant chỉ tồn tại với matrix vuông."
            )

        return np.linalg.det(self.data)

    # ==========================================
    # INVERSE
    # ==========================================

    def inverse(self):
        if not self.is_square():
            raise ValueError(
                "Chỉ matrix vuông mới có inverse."
            )

        if self.is_singular():
            raise ValueError(
                "Matrix suy biến, không có inverse."
            )

        return Matrix(
            np.linalg.inv(self.data)
        )

    # ==========================================
    # RANK
    # ==========================================

    def rank(self):
        return np.linalg.matrix_rank(self.data)

    # ==========================================
    # NORM
    # ==========================================

    def norm(self):
        return np.linalg.norm(self.data)

    # ==========================================
    # EIGENVALUES
    # ==========================================

    def eigenvalues(self):
        if not self.is_square():
            raise ValueError(
                "Eigenvalues yêu cầu matrix vuông."
            )

        return np.linalg.eigvals(self.data)

    # ==========================================
    # EIGENVECTORS
    # ==========================================

    def eigenvectors(self):
        if not self.is_square():
            raise ValueError(
                "Eigenvectors yêu cầu matrix vuông."
            )

        eigenvalues, eigenvectors = np.linalg.eig(
            self.data
        )

        return eigenvalues, eigenvectors

    # ==========================================
    # POSITIVE DEFINITE
    # ==========================================

    def is_positive_definite(self):
        """
        A positive definite iff:
            x^T A x > 0
        for every x != 0.

        Với matrix đối xứng:
            A positive definite
            <=> tất cả eigenvalues > 0
            <=> Cholesky decomposition tồn tại.
        """

        if not self.is_square():
            return False

        if not self.is_symmetric():
            return False

        try:
            np.linalg.cholesky(self.data)
            return True
        except np.linalg.LinAlgError:
            return False

    # ==========================================
    # SPECTRAL DECOMPOSITION
    # ==========================================

    def spectral_decomposition(self):
        """
        Với symmetric matrix:

            A = Q Λ Q^T

        Q: eigenvectors
        Λ: diagonal eigenvalues
        """

        if not self.is_square():
            raise ValueError(
                "Spectral decomposition yêu cầu matrix vuông."
            )

        if not self.is_symmetric():
            raise ValueError(
                "Spectral decomposition dạng "
                "QΛQ^T yêu cầu matrix symmetric."
            )

        eigenvalues, eigenvectors = np.linalg.eigh(
            self.data
        )

        Lambda = Matrix(
            np.diag(eigenvalues)
        )

        Q = Matrix(
            eigenvectors
        )

        return Q, Lambda

    # ==========================================
    # COPY
    # ==========================================

    def copy(self):
        return Matrix(
            self.data.copy()
        )

    # ==========================================
    # NUMPY
    # ==========================================

    def to_numpy(self):
        return self.data.copy()

    # ==========================================
    # INDEX
    # ==========================================

    def __getitem__(self, index):
        return self.data[index]

    # ==========================================
    # ADDITION
    # ==========================================

    def __add__(self, other):
        from .operations import MatrixOperations

        return MatrixOperations.add(
            self,
            other
        )

    # ==========================================
    # SUBTRACTION
    # ==========================================

    def __sub__(self, other):
        from .operations import MatrixOperations

        return MatrixOperations.subtract(
            self,
            other
        )

    # ==========================================
    # MULTIPLICATION
    # ==========================================

    def __matmul__(self, other):
        from .operations import MatrixOperations

        return MatrixOperations.multiply(
            self,
            other
        )

    # ==========================================
    # SCALAR MULTIPLICATION
    # ==========================================

    def __mul__(self, other):

        if isinstance(other, (int, float)):

            from .operations import MatrixOperations

            return MatrixOperations.scalar_multiply(
                self,
                other
            )

        raise TypeError(
            "Matrix * chỉ hỗ trợ scalar. "
            "Dùng @ cho matrix multiplication."
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    # ==========================================
    # REPRESENTATION
    # ==========================================

    def __repr__(self):
        return f"Matrix(\n{self.data}\n)"