import numpy as np

from .base import Matrix


class MatrixAnalysis:

    def __init__(self, A: Matrix):
        if not isinstance(A, Matrix):
            raise TypeError(
                "A phải là Matrix."
            )

        self.A = A

    # ==========================================
    # a. SYMMETRIC
    # ==========================================

    def is_symmetric(self):
        return self.A.is_symmetric()

    # ==========================================
    # b. POSITIVE DEFINITE
    # ==========================================

    def is_positive_definite(self):
        return self.A.is_positive_definite()

    def positive_definite_proof(self):
        """
        Trả về thông tin để chứng minh
        positive definite bằng eigenvalues.
        """

        eigenvalues = np.linalg.eigvalsh(
            self.A.data
        )

        return {
            "eigenvalues": eigenvalues,
            "all_positive": np.all(eigenvalues > 0)
        }

    # ==========================================
    # c. EIGENVALUES / EIGENVECTORS
    # ==========================================

    def eigen(self):
        """
        Sử dụng eigh cho symmetric matrix.
        """

        if self.A.is_symmetric():

            eigenvalues, eigenvectors = np.linalg.eigh(
                self.A.data
            )

        else:

            eigenvalues, eigenvectors = np.linalg.eig(
                self.A.data
            )

        return eigenvalues, eigenvectors

    # ==========================================
    # d. SPECTRAL DECOMPOSITION
    # ==========================================

    def spectral_decomposition(self):

        if not self.A.is_symmetric():
            raise ValueError(
                "Spectral decomposition QΛQᵀ "
                "yêu cầu matrix symmetric."
            )

        eigenvalues, eigenvectors = np.linalg.eigh(
            self.A.data
        )

        Q = Matrix(eigenvectors)

        Lambda = Matrix(
            np.diag(eigenvalues)
        )

        return Q, Lambda

    # ==========================================
    # e. INVERSE
    # ==========================================

    def inverse(self):

        return self.A.inverse()

    # ==========================================
    # f. EIGEN OF INVERSE
    # ==========================================

    def inverse_eigen(self):

        A_inverse = self.A.inverse()

        if A_inverse.is_symmetric():

            eigenvalues, eigenvectors = np.linalg.eigh(
                A_inverse.data
            )

        else:

            eigenvalues, eigenvectors = np.linalg.eig(
                A_inverse.data
            )

        return eigenvalues, eigenvectors

    # ==========================================
    # COMPLETE ANALYSIS
    # ==========================================

    def analyze(self):

        result = {}

        # a
        result["symmetric"] = self.is_symmetric()

        # b
        result["positive_definite"] = (
            self.is_positive_definite()
        )

        result["positive_definite_proof"] = (
            self.positive_definite_proof()
        )

        # c
        eigenvalues, eigenvectors = self.eigen()

        result["eigenvalues"] = eigenvalues
        result["eigenvectors"] = eigenvectors

        # d
        if self.A.is_symmetric():

            Q, Lambda = (
                self.spectral_decomposition()
            )

            result["Q"] = Q
            result["Lambda"] = Lambda

        # e
        A_inverse = self.inverse()

        result["inverse"] = A_inverse

        # f
        inverse_eigenvalues, inverse_eigenvectors = (
            self.inverse_eigen()
        )

        result["inverse_eigenvalues"] = (
            inverse_eigenvalues
        )

        result["inverse_eigenvectors"] = (
            inverse_eigenvectors
        )

        return result