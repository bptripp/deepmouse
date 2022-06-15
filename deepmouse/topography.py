import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.maps.flatmap import FlatMap
from deepmouse.maps.map import right_target_indices, get_positions

"""
This code assigns coordinates to each voxel of each primary sensory area based on its flatmap
position. It propagates these coordinates through the connectivity model. This produces, for each
non-primary voxel, an estimate of its coordinates in sensory space.
"""

# TODO: deconvolve mesoscale model smoothing?

cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()


class NormalizedFlatmap:
    def __init__(self, area):
        flatmap = FlatMap(area=area)
        flatmap._fit()
        positions_2d = flatmap.positions_2d.T

        centre = np.mean(positions_2d, axis=0)
        squared_distances = np.sum((positions_2d - centre)**2, axis=1)
        sd = (np.sum(squared_distances)/len(positions_2d))**.5

        rel_positions = positions_2d - centre
        self.positions_2d = rel_positions / sd
        self.positions_3d = flatmap.positions_3d.T

    def get_position(self, index):
        return self.rel_positions[index,:]

    def get_position(self, position_3d):
        index = np.where((self.positions_3d == position_3d).all(axis=1))[0]
        if len(index) == 0:
            raise Exception('Unknown voxel: {}'.format(position_3d))
        return self.positions_2d[index]


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
    def __init__(self):
        self.gaussians = []

    def add(self, gaussian):
        self.gaussians.append(gaussian)

    def approx(self):
        """
        :return: Gaussian2D with same mean and covariance as the mixture.
        """
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


def get_primary_gaussians(area):
    positions_3d = get_positions(cache, area)
    flatmap = NormalizedFlatmap(area)
    gaussians = []
    for position_3d in positions_3d:
        mean = flatmap.get_position(position_3d)
        covariance = [[0, 0], [0, 0]] # primary voxels have no spread in primary voxel space
        gaussians.append(Gaussian2D(1, mean, covariance))

    return gaussians, positions_3d


def propagate_gaussians_through_isocortex(gaussians, positions_3d, data_folder='data_files/'):
    """
    :param gaussians: Gaussian model of RF for selected isocortex voxels
    :param positions_3d: 3d positions of these voxels
    :return: Gaussian models of input to every target voxel in right isocortex
    """

    weight_file = data_folder + '/voxel-weights.pkl'
    node_file = data_folder + '/voxel-nodes.pkl'
    if os.path.isfile(weight_file) and os.path.isfile(node_file):
        with open(weight_file, 'rb') as file:
            weights = pickle.load(file)
        with open(node_file, 'rb') as file:
            nodes = pickle.load(file)
    else:
        raise Exception('Weight files missing')

    print(weights.shape) # source to latent (226346, 428)
    print(nodes.shape) # latent to target (both hemispheres) (428, 448962)

    cortex_id = structure_tree.get_id_acronym_map()['Isocortex']

    source_mask = cache.get_source_mask()
    source_keys = source_mask.get_key(structure_ids=None) # structure ids for all voxels
    print('len source keys {}'.format(len(source_keys)))

    # indices of source cortex voxels among all voxels
    source_cortex_indices = []
    for i in range(len(source_keys)):
        if structure_tree.structure_descends_from(source_keys[i], cortex_id):
            source_cortex_indices.append(i)

    target_mask = cache.get_target_mask()
    target_keys = target_mask.get_key(structure_ids=None) # structure ids for all voxels
    print('len target keys {}'.format(len(target_keys)))

    target_cortex_indices = []
    for i in range(len(target_keys)):
        if structure_tree.structure_descends_from(target_keys[i], cortex_id):
            target_cortex_indices.append(i)

    #right_target_cortex_indices will not be same as source_cortex_indices but positions in same order
    target_cortex_indices = np.array(target_cortex_indices)
    r = right_target_indices(cache)
    right_target_cortex_indices = target_cortex_indices[r]

    cortex_weights = weights[source_cortex_indices,:]
    positions_all = get_positions(cache, 'Isocortex')  # these are in source order

    def get_source_index(position_3d):
        ind = np.where((positions_all == position_3d).all(axis=1))[0]
        if len(ind) == 0:
            raise Exception('Unknown voxel: {}'.format(position_3d))
        return ind

    indices = [get_source_index(p) for p in positions_3d]

    def get_mixture_for_target_voxel(target_index):
        target_weights = np.dot(nodes[:,target_index].T, cortex_weights.T)
        inclusion_threshold = 0.01 * np.max(target_weights)
        mixture = GaussianMixture2D()
        for g, p, ind in zip(gaussians, positions_3d, indices):
            weight = target_weights[ind]
            if weight > inclusion_threshold:
                mixture.add(Gaussian2D(weight*g.weight, g.mean, g.covariance))

        # print('{} of {} th: {}'.format(len(mixture.gaussians), len(gaussians), inclusion_threshold))
        return mixture

    result = []
    for i, target_index in enumerate(right_target_cortex_indices):
        if i % 100 == 0:
            print('{} of {}'.format(i, len(right_target_cortex_indices)))

        mixture = get_mixture_for_target_voxel(target_index)

        result.append(mixture.approx())
    return result


if __name__ == '__main__':
    # g1 = Gaussian2D(1, [2, 0], [[4, 1], [1, 4]])
    # g2 = Gaussian2D(1, [0, 0], [[4, 1], [1, 4]])
    # mix = GaussianMixture2D([g1, g2])
    # print(mix.approx())

    # # area = 'VISp'
    # # area = 'AUDp'
    # # area = 'PIR'
    # # area = 'SSp-bfd'
    # # area = 'SSp-m'
    # # area = 'SSp-n'
    # area = 'SSp-ul'
    area = 'SSp-n'

    gaussians, positions_3d = get_primary_gaussians(area)
    propagated = propagate_gaussians_through_isocortex(gaussians, positions_3d)

    with open('generated/propagated {}'.format(area), 'wb') as file:
        pickle.dump(propagated, file)

    with open('generated/propagated {}'.format(area), 'rb') as file:
        propagated = pickle.load(file)

    from deepmouse.geodesic_flatmap import GeodesicFlatmap
    flatmap = GeodesicFlatmap()

    weights = [gaussian.weight for gaussian in propagated]
    max_weight = np.max(weights)
    print(max_weight)

    # set voxel colours
    for i, gaussian in enumerate(propagated):
        if len(gaussian.mean) < 2: # bug in gaussian, change this
            x = gaussian.mean[0][0]
            x = np.clip(x, -1.5, 1.5)
            red = (x+1.5)/3
            blue = 1-red
            brightness = (gaussian.weight / max_weight) ** (1/4)
            flatmap.set_voxel_colour(flatmap.voxel_positions[i], [brightness*red, 0, brightness*blue])

    flatmap.show_map(image_file='generated/bar.png')
    flatmap.draw_boundary('VISp')
    flatmap.draw_boundary('AUDp')
    flatmap.draw_boundary('SSp-bfd')
    plt.savefig('generated/{}-ml.png'.format(area))
    plt.show()


