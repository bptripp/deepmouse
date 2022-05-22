import numpy as np
from deepmouse.maps.util import get_voxel_model_cache, get_positions, get_id, get_default_structure_tree
from deepmouse.geodesic_flatmap import GeodesicFlatmap
from deepmouse.maps.flatmap import FlatMap
"""
This code assigns coordinates to each voxel of each primary sensory area based on its flatmap
position. It propagates these coordinates through the connectivity model. This produces, for each
non-primary voxel, an estimate of its coordinates in sensory space.
"""

# # area = 'VISp'
# # area = 'AUDp'
# # area = 'PIR'
# # area = 'SSp-bfd'
# # area = 'SSp-m'
# # area = 'SSp-n'
# area = 'SSp-ul'
# flatmap = FlatMap(area=area)
# flatmap._fit()
# positions_2d = flatmap.positions_2d.T
#
# centre = np.mean(positions_2d, axis=0)
# squared_distances = np.sum((positions_2d - centre)**2, axis=1)
# sd = (np.sum(squared_distances)/len(positions_2d))**.5
# print(centre)
# print(sd)
#
# rel_positions = positions_2d - centre
# rel_positions = rel_positions / sd
#
# import matplotlib.pyplot as plt
# plt.scatter(rel_positions[:,0], rel_positions[:,1])
# # plt.scatter(positions_2d[:,0], positions_2d[:,1])
# plt.axis('equal')
# plt.show()


class Gaussian2D:
    def __init__(self, weight, mean, covariance):
        self.weight = weight
        self.mean = np.array(mean)
        self.covariance = np.array(covariance)

    def to_vector(self):
        np.concatenate(self.weight, self.mean.flatten(), self.covariance.flatten())

    def __str__(self):
        return 'weight: {} mean: {} covariance: {}'.format(self.weight, self.mean, self.covariance.flatten())


class GaussianMixture2D:
    def __init__(self, gaussians):
        self.gaussians = gaussians

    def approx(self):
        weight_sum = 0
        weighted_sum_means = np.zeros(2)
        covariance = np.zeros((2,2))

        for g in self.gaussians:
            weight_sum = weight_sum + g.weight
            weighted_sum_means = weighted_sum_means + g.weight * g.mean
        mean = weighted_sum_means / weight_sum

        for g in self.gaussians:
            between = np.outer(g.mean - mean, g.mean - mean)
            covariance = covariance + g.weight/weight_sum*(between + g.covariance)

        return Gaussian2D(weight_sum, mean, covariance)


if __name__ == '__main__':
    g1 = Gaussian2D(1, [2, 0], [[4, 1], [1, 4]])
    g2 = Gaussian2D(1, [0, 0], [[4, 1], [1, 4]])
    mix = GaussianMixture2D([g1, g2])
    print(mix.approx())
