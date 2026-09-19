import numpy as np

from .base import Matrix


class MatrixOperations:

    # ==========================================
    # Addition
    # ==========================================

    @staticmethod
    def add(A: Matrix, B: Matrix):
        """
        A + B
        """

        if A.shape != B.shape:
            raise ValueError(
                f"Không thể cộng matrix "
                f"{A.shape} và {B.shape}."
            )

        return Matrix(
            A.data + B.data
        )

    # ==========================================
    # Subtraction
    # ==========================================

    @staticmethod
    def subtract(A: Matrix, B: Matrix):
        """
        A - B
        """

        if A.shape != B.shape:
            raise ValueError(
                f"Không thể trừ matrix "
                f"{A.shape} và {B.shape}."
            )

        return Matrix(
            A.data - B.data
        )

    # ==========================================
    # Scalar multiplication
    # ==========================================

    @staticmethod
    def scalar_multiply(
        A: Matrix,
        scalar: float
    ):
        """
        kA
        """

        return Matrix(
            A.data * scalar
        )

    # ==========================================
    # Matrix multiplication
    # ==========================================

    @staticmethod
    def multiply(A: Matrix, B: Matrix):
        """
        Matrix multiplication:

            C = AB

        Điều kiện:

            A(m x n)
            B(n x p)

            => C(m x p)
        """

        if A.columns != B.rows:
            raise ValueError(
                f"Không thể nhân "
                f"{A.shape} × {B.shape}."
            )

        return Matrix(
            A.data @ B.data
        )

    # ==========================================
    # Element-wise multiplication
    # ==========================================

    @staticmethod
    def elementwise_multiply(
        A: Matrix,
        B: Matrix
    ):
        """
        Nhân từng phần tử:

            C_ij = A_ij * B_ij
        """

        if A.shape != B.shape:
            raise ValueError(
                "Hai matrix phải cùng kích thước."
            )

        return Matrix(
            A.data * B.data
        )

    # ==========================================
    # Element-wise division
    # ==========================================

    @staticmethod
    def elementwise_divide(
        A: Matrix,
        B: Matrix
    ):
        """
        Chia từng phần tử.
        """

        if A.shape != B.shape:
            raise ValueError(
                "Hai matrix phải cùng kích thước."
            )

        if np.any(B.data == 0):
            raise ZeroDivisionError(
                "Không thể chia cho 0."
            )

        return Matrix(
            A.data / B.data
        )

    # ==========================================
    # Matrix power
    # ==========================================

    @staticmethod
    def power(
        A: Matrix,
        exponent: int
    ):
        """
        A^n
        """

        if not A.is_square():
            raise ValueError(
                "Matrix power yêu cầu ma trận vuông."
            )

        return Matrix(
            np.linalg.matrix_power(
                A.data,
                exponent
            )
        )

    # ==========================================
    # Matrix-vector multiplication
    # ==========================================

    @staticmethod
    def multiply_vector(A, v):
        """
        A × v
        """

        if A.columns != v.dimension:
            raise ValueError(
                "Số cột của Matrix phải "
                "bằng số chiều của Vector."
            )

        from .vector import Vector

        result = A.data @ v.values

        return Vector(result)