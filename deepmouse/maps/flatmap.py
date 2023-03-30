import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from mcmodels.core import VoxelModelCache
from deepmouse.maps.util import get_id, get_default_structure_tree
# from maps.util import get_id, get_default_structure_tree


def get_angles(offset):
    """
    :param offset: voxel position relative to centre (forward, up, right)
    :return: [angle to right from sagittal plane, angle forward from up in sagittal plane]
    """
    return np.array(
        [
            np.arctan(offset[2] / np.linalg.norm(offset[:2])),
            np.arctan2(offset[1], offset[0]),
        ]
    )


class FlatMap:
    _instance = {"visual": None, "barrel": None}

    def __init__(self, area="visual"):
        """
        A simple flat map of a single mouse cortical area.
        """

        if area == "visual":  # for backward compatibility
            structure_id = 669
        else:
            structure_tree = get_default_structure_tree()
            structure_id = get_id(structure_tree, area)

        cache = VoxelModelCache(manifest_file="connectivity/voxel_model_manifest.json")
        source_mask = cache.get_source_mask()

        source_keys = source_mask.get_key(structure_ids=[structure_id])
        self.source_key_volume = source_mask.map_masked_to_annotation(source_keys)
        self.positions_3d = np.array(np.nonzero(self.source_key_volume))

        self._positions_3d_for_fit = self.positions_3d
        if not (area == "visual" or area == "PIR"):
            structure_id_for_fit = get_id(
                structure_tree, "{}5".format(area)
            )  # fit using just L5
            source_keys_for_fit = source_mask.get_key(
                structure_ids=[structure_id_for_fit]
            )
            source_key_volume_for_fit = source_mask.map_masked_to_annotation(
                source_keys_for_fit
            )
            self._positions_3d_for_fit = np.array(np.nonzero(source_key_volume_for_fit))

        self.voxel_size = 0.1  # mm

    def _fit(self):
        # fit sphere surface to cortex voxels

        def fun(x):
            # x = [x centre, y centre, z centre, radius]
            centre = x[:3]
            radius = x[3]
            offsets = self._positions_3d_for_fit.T - centre
            distances = np.linalg.norm(offsets, axis=1)
            return distances - radius

        res_lsq = least_squares(fun, [50, 10, 50, 50])
        self.centre = res_lsq.x[:3]
        self.radius = res_lsq.x[3]

        mean_offset = (
            np.mean(self.positions_3d.T, axis=0) - self.centre
        )  # back, down, right
        mean_offset[:2] = -mean_offset[:2]  # forward, up, right
        a0, a1 = get_angles(mean_offset)

        # finding rotation from vertical to mean_offset (mean_offset = R2*R1*vertical)
        a1 = a1 - np.pi / 2
        R1 = np.array(
            [[1, 0, 0], [0, np.cos(a0), -np.sin(a0)], [0, np.sin(a0), np.cos(a0)]]
        )
        R2 = np.array(
            [[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]]
        )
        R = np.matmul(R2, R1)
        self.R_vertical = np.linalg.inv(R)  # this rotates mean offset to top

        n = self.positions_3d.shape[1]
        self.positions_2d = np.zeros((2, n))
        for i in range(n):
            self.positions_2d[:, i] = self.get_position_2d(self.positions_3d[:, i])

        return self.centre, self.radius

    def _plot_residuals(self):
        offsets = self.positions_3d.T - self.centre
        distances = np.linalg.norm(offsets, axis=1)
        residuals = distances - self.radius
        plt.hist(residuals)
        plt.title("cortex is about 9 voxels thick")
        plt.xlabel("voxel distances from projection surface")
        plt.ylabel("frequency")
        plt.show()

    def get_position_2d(self, position_3d):
        """
        :param position_3d: 3D voxel position (voxels)
        :return: 2D voxel position (mm along surface)
        """
        offset = position_3d.T - self.centre
        offset[:2] = -offset[:2]  # forward, up, right
        offset = np.matmul(self.R_vertical, offset)

        angles = get_angles(offset)
        angles[1] = angles[1] - np.pi / 2
        millimeters = self.voxel_size * self.radius * angles

        return [millimeters[0], -millimeters[1]]

    def _plot_voxels(self):
        # takes about 30s
        voxels = self.source_key_volume > 0
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.voxels(voxels, facecolors=[1, 1, 1, 0.5], edgecolor="k")
        ax.set_xlabel("back")
        ax.set_ylabel("down")
        ax.set_zlabel("lateral")
        ax.set_xlim((min(self.positions_3d[0, :]), max(self.positions_3d[0, :])))
        ax.set_ylim((min(self.positions_3d[1, :]), max(self.positions_3d[1, :])))
        ax.set_zlim((min(self.positions_3d[2, :]), max(self.positions_3d[2, :])))
        plt.show()

    @staticmethod
    def get_instance(area="visual"):
        """
        :return: Shared instance of FlatMap
        """
        if FlatMap._instance[area] is None:
            FlatMap._instance[area] = FlatMap(area=area)
            FlatMap._instance[area]._fit()
        return FlatMap._instance[area]


if __name__ == "__main__":
    flatmap = FlatMap.get_instance(area="visual")
    flatmap._plot_voxels()
    centre, radius = flatmap._fit()
    print("centre: {} radius: {}".format(centre, radius))
    flatmap._plot_residuals()
