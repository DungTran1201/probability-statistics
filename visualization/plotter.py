import numpy as np
import matplotlib.pyplot as plt

from core.vector import Vector
from core.relation import VectorRelation


class VectorPlotter:

    def __init__(self):
        pass

    def plot_2d(
        self,
        vector1: Vector,
        vector2: Vector,
        show_projection=True,
        show_angle=True
    ):
        """
        Vẽ hai vector trong không gian 2D.
        """

        self._check_dimension(vector1, 2)
        self._check_dimension(vector2, 2)

        relation = VectorRelation(
            vector1,
            vector2
        )

        fig, ax = plt.subplots(
            figsize=(9, 9)
        )

        v1 = vector1.values
        v2 = vector2.values

        # Vector 1
        ax.quiver(
            0,
            0,
            v1[0],
            v1[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
            label="Vector 1"
        )

        # Vector 2
        ax.quiver(
            0,
            0,
            v2[0],
            v2[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="blue",
            label="Vector 2"
        )

        if show_projection:

            projection = (
                relation
                .projection_v1_on_v2()
            )

            p = projection.values

            # Vector projection
            ax.quiver(
                0,
                0,
                p[0],
                p[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="green",
                linestyle="--",
                label="Projection"
            )

            # Đường vuông góc
            ax.plot(
                [v1[0], p[0]],
                [v1[1], p[1]],
                linestyle="--",
                color="gray"
            )

        # Giới hạn trục
        max_value = max(
            np.max(np.abs(v1)),
            np.max(np.abs(v2))
        ) + 2

        ax.set_xlim(-max_value, max_value)
        ax.set_ylim(-max_value, max_value)

        ax.axhline(
            0,
            color="black",
            linewidth=0.8
        )

        ax.axvline(
            0,
            color="black",
            linewidth=0.8
        )

        ax.grid(True)
        ax.set_aspect("equal")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.set_title(
            f"Angle = "
            f"{relation.angle_degrees():.2f}°"
        )

        ax.legend()

        plt.show()

    def plot_3d(
        self,
        vector1: Vector,
        vector2: Vector,
        show_projection=True
    ):
        """
        Vẽ hai vector trong không gian 3D.
        """

        self._check_dimension(vector1, 3)
        self._check_dimension(vector2, 3)

        relation = VectorRelation(
            vector1,
            vector2
        )

        fig = plt.figure(
            figsize=(10, 8)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        v1 = vector1.values
        v2 = vector2.values

        # Vector 1
        ax.quiver(
            0, 0, 0,
            v1[0],
            v1[1],
            v1[2],
            color="red",
            arrow_length_ratio=0.1,
            label="Vector 1"
        )

        # Vector 2
        ax.quiver(
            0, 0, 0,
            v2[0],
            v2[1],
            v2[2],
            color="blue",
            arrow_length_ratio=0.1,
            label="Vector 2"
        )

        if show_projection:

            projection = (
                relation
                .projection_v1_on_v2()
            )

            p = projection.values

            # Projection
            ax.quiver(
                0, 0, 0,
                p[0],
                p[1],
                p[2],
                color="green",
                arrow_length_ratio=0.1,
                linestyle="--",
                label="Projection"
            )

            # Đường vuông góc
            ax.plot(
                [v1[0], p[0]],
                [v1[1], p[1]],
                [v1[2], p[2]],
                linestyle="--",
                color="gray"
            )

        max_value = max(
            np.max(np.abs(v1)),
            np.max(np.abs(v2))
        ) + 2

        ax.set_xlim(
            -max_value,
            max_value
        )

        ax.set_ylim(
            -max_value,
            max_value
        )

        ax.set_zlim(
            -max_value,
            max_value
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title(
            f"Angle = "
            f"{relation.angle_degrees():.2f}°"
        )

        ax.legend()

        plt.show()

    @staticmethod
    def _check_dimension(
        vector: Vector,
        dimension: int
    ):
        if vector.dimension != dimension:
            raise ValueError(
                f"Vector phải là vector "
                f"{dimension}D."
            )