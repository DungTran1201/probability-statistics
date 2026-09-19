import math
import numpy as np

from .vector import Vector


class VectorRelation:

    def __init__(self, vector1: Vector, vector2: Vector):
        vector1._check_dimension(vector2)

        self.v1 = vector1
        self.v2 = vector2

    def dot_product(self):
        """
        Tích vô hướng:

            a · b = |a||b|cos(theta)
        """
        return self.v1.dot(self.v2)

    def angle_radians(self):
        """
        Góc giữa hai vector theo radian.
        """

        magnitude1 = self.v1.magnitude()
        magnitude2 = self.v2.magnitude()

        if magnitude1 == 0 or magnitude2 == 0:
            raise ValueError(
                "Không xác định góc với vector 0."
            )

        cos_theta = (
            self.dot_product()
            / (magnitude1 * magnitude2)
        )

        # Tránh sai số floating point
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return math.acos(cos_theta)

    def angle_degrees(self):
        """
        Góc giữa hai vector theo degree.
        """

        return math.degrees(
            self.angle_radians()
        )

    def projection_v1_on_v2(self):
        """
        Hình chiếu vector v1 lên v2:

             proj_v2(v1)
             = (v1 · v2 / ||v2||²) v2
        """

        denominator = self.v2.dot(self.v2)

        if denominator == 0:
            raise ValueError(
                "Không thể chiếu lên vector 0."
            )

        coefficient = (
            self.v1.dot(self.v2)
            / denominator
        )

        return self.v2.multiply(coefficient)

    def projection_v2_on_v1(self):
        """
        Hình chiếu vector v2 lên v1.
        """

        denominator = self.v1.dot(self.v1)

        if denominator == 0:
            raise ValueError(
                "Không thể chiếu lên vector 0."
            )

        coefficient = (
            self.v2.dot(self.v1)
            / denominator
        )

        return self.v1.multiply(coefficient)

    def perpendicular_component_v1(self):
        """
        Thành phần vuông góc của v1 đối với v2:

            v1_perp = v1 - proj_v2(v1)
        """

        projection = self.projection_v1_on_v2()

        return self.v1.subtract(projection)

    def is_orthogonal(self, tolerance=1e-9):
        """
        Kiểm tra hai vector có vuông góc hay không.

        a · b = 0
        """

        return abs(self.dot_product()) < tolerance

    def is_parallel(self, tolerance=1e-9):
        """
        Kiểm tra hai vector có song song hay không.
        """

        angle = self.angle_degrees()

        return (
            abs(angle) < tolerance
            or abs(angle - 180) < tolerance
        )

    def summary(self):
        """
        Tổng hợp quan hệ giữa hai vector.
        """

        return {
            "dot_product": self.dot_product(),
            "angle_degrees": self.angle_degrees(),
            "projection_v1_on_v2":
                self.projection_v1_on_v2(),
            "projection_v2_on_v1":
                self.projection_v2_on_v1(),
            "perpendicular_component_v1":
                self.perpendicular_component_v1(),
            "orthogonal":
                self.is_orthogonal(),
            "parallel":
                self.is_parallel(),
        }