import numpy as np


class Vector:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

        if self.values.ndim != 1:
            raise ValueError("Vector phải là mảng 1 chiều.")

    @property
    def dimension(self):
        return self.values.shape[0]

    def magnitude(self):
        return np.linalg.norm(self.values)

    def normalize(self):
        magnitude = self.magnitude()

        if np.isclose(magnitude, 0):
            raise ValueError("Không thể normalize vector 0.")

        return Vector(self.values / magnitude)

    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Dot product yêu cầu một Vector.")

        if self.dimension != other.dimension:
            raise ValueError(
                f"Không thể dot product Vector "
                f"{self.dimension}D và {other.dimension}D."
            )

        return float(np.dot(self.values, other.values))

    def add(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Hai vector phải cùng số chiều.")

        return Vector(self.values + other.values)

    def subtract(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Hai vector phải cùng số chiều.")

        return Vector(self.values - other.values)

    def scalar_multiply(self, scalar):
        return Vector(self.values * scalar)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                "Vector * chỉ hỗ trợ scalar."
            )

        return self.scalar_multiply(scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __matmul__(self, other):
        """
        Vector @ Vector → scalar
        Vector @ Matrix → Vector
        """

        # Vector × Vector
        if isinstance(other, Vector):
            return self.dot(other)

        # Vector × Matrix
        from .matrix import Matrix

        if isinstance(other, Matrix):
            if self.dimension != other.rows:
                raise ValueError(
                    f"Không thể nhân Vector {self.dimension}D "
                    f"với Matrix {other.shape}."
                )

            result = self.values @ other.data

            return Vector(result)

        raise TypeError(
            "Vector @ chỉ hỗ trợ Vector hoặc Matrix."
        )

    def __repr__(self):
        return f"Vector({self.values})"