import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
# from sklearn.manifold import locally_linear_embedding
from mcmodels.core import VoxelModelCache


class FlatMap:
    def __init__(self):
        """
        Creates a flat-map of mouse visual cortex based on manifold learning.

        :param voxels:
        """
        cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
        source_mask = cache.get_source_mask()
        source_keys = source_mask.get_key(structure_ids=[669]) #669: visual areas; 913: visual areas L4 doesn't work
        self.source_key_volume = source_mask.map_masked_to_annotation(source_keys)
        self.positions_3d = np.array(np.nonzero(self.source_key_volume))

        # pos, err = locally_linear_embedding(idx, n_neighbors=50, n_components=2)

    def _fit(self):
        # fit sphere surface to cortex voxels

        def fun(x):
            # x = [x centre, y centre, z centre, radius]
            centre = x[:3]
            radius = x[3]
            offsets = self.positions_3d.T - centre
            distances = np.linalg.norm(offsets, axis=1)
            return distances - radius

        res_lsq = least_squares(fun, [50, 50, 50, 50])
        centre = res_lsq.x[:3]
        radius = res_lsq.x[3]
        self.centre = centre

        n = self.positions_3d.shape[1]
        self.positions_2d = np.zeros((2, n))
        for i in range(n):
            self.positions_2d[:,i] = self.get_position_2d(self.positions_3d[:,i])

        # offsets = self.positions_3d.T - centre
        # for i in range(n):
        #     self.positions_2d[1,i] = np.arctan2(-offsets[i,0], -offsets[i,1])
        #     self.positions_2d[0,i] = np.arctan(offsets[i,2]/np.linalg.norm(offsets[i,:2]))

        # # looks reasonable
        # distances = np.linalg.norm(offsets, axis=1)
        # errors = distances - radius
        # plt.hist(errors)
        # plt.show()

        return centre, radius

    def get_position_2d(self, position_3d):
        """
        :param position_3d:
        :return:
        """
        offset = position_3d.T - self.centre

        result = np.zeros(2)
        result[1] = np.arctan2(-offset[0], -offset[1])
        result[0] = np.arctan(offset[2]/np.linalg.norm(offset[:2]))
        return result

    def _save(self):
        pass

    def _plot_voxels(self):
        # takes about 30s
        voxels = self.source_key_volume > 0
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=[1, 1, 1, 0.5], edgecolor='k')
        ax.set_xlabel('back')
        ax.set_ylabel('down')
        ax.set_zlabel('lateral')
        ax.set_xlim((min(self.positions_3d[0,:]), max(self.positions_3d[0,:])))
        ax.set_ylim((min(self.positions_3d[1,:]), max(self.positions_3d[1,:])))
        ax.set_zlim((min(self.positions_3d[2,:]), max(self.positions_3d[2,:])))
        plt.show()

    def plot_voxels(self, positions):
        # TODO: this doesn't really belong here
        voxels = self.source_key_volume < 0 #all False
        print(voxels.shape)
        print(positions)
        voxels[positions[0], positions[1], positions[2]] = True
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=[1, 1, 1, 0.5], edgecolor='k')
        ax.set_xlabel('back')
        ax.set_ylabel('down')
        ax.set_zlabel('lateral')
        ax.set_xlim((min(self.positions_3d[0,:]), max(self.positions_3d[0,:])))
        ax.set_ylim((min(self.positions_3d[1,:]), max(self.positions_3d[1,:])))
        ax.set_zlim((min(self.positions_3d[2,:]), max(self.positions_3d[2,:])))
        plt.show()


if __name__ == '__main__':
    flatmap = FlatMap()
    # flatmap._plot_voxels()
    centre, radius = flatmap._fit()

    flatmap.plot_voxels(flatmap.positions_3d)

    # plt.scatter(flatmap.positions_2d[0,:], flatmap.positions_2d[1,:])
    # plt.show()
    # print(centre)
    # print(radius)
