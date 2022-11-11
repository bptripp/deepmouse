import numpy as np
import pickle

from argparse import ArgumentParser
from maps.util import get_voxel_model_cache, get_default_structure_tree
from maps.map import right_target_indices, get_positions
from maps.flatmap import FlatMap
from streamlines import CompositeInterpolator
from horizontal_distance import shortest_distance, surface_to_surface_streamline

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--omit_experiment_rank","-o",default=None)

    return parser.parse_args()

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

        if weight_sum > 0:
            mean = weighted_sum_means / weight_sum

            for g in self.gaussians:
                between = np.outer(g.mean - mean, g.mean - mean)
                covariance = covariance + g.weight/weight_sum*(between + g.covariance)
        else:
            mean = [0, 0]
            covariance = [[0, 0], [0, 0]]

        return Gaussian2D(weight_sum, mean, covariance)

def get_primary_gaussians(cache,area):
    positions_3d = get_positions(cache, area)
    flatmap = NormalizedFlatmap(area)
    gaussians = []
    for position_3d in positions_3d:
        mean = flatmap.get_position(position_3d).squeeze()
        covariance = [[0, 0], [0, 0]] # primary voxels have no spread in primary voxel space
        gaussians.append(Gaussian2D(1, mean, covariance))

    return gaussians, positions_3d

def remove_experiment(weights, nodes, index):
    num_experiments = weights.shape[1]
    if index >= num_experiments:
        raise Exception('Index {} given but only {} experiments'.format(index, num_experiments))

    new_weights = np.concatenate((weights[:, :index], weights[:, index + 1:]), axis=1)
    new_nodes = np.concatenate((nodes[:index, :], nodes[index+1:, :]), axis=0)

    # re-normalize weights (transposes are to make broadcasting work)
    new_weights = (new_weights.T / np.sum(new_weights, 1).T).T

    return new_weights, new_nodes

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Instantiate voxel model cache and structure tree
    cache = get_voxel_model_cache()
    structure_tree = get_default_structure_tree()

    # Areas to simulate - can be changed as needed
    areas = ['VISp', 'AUDp', 'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-n']


    # Propagate gaussians for each area
    for area in areas:

        # Get gaussians and positions of the voxels in the area
        gaussians, positions_3d = get_primary_gaussians(cache,area)

        # Weight files are massive, using randomly generated weights for the moment
        # as they do not affect functionality

        # weights = cache.get_weights()
        # nodes = cache.get_nodes()
        weights = np.random.rand(226346, 428)
        nodes = np.random.rand(428, 448962)

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
        # Find indices for right isocortex
        r = right_target_indices(cache)
        right_target_cortex_indices = target_cortex_indices[r]

        cortex_weights = weights[source_cortex_indices,:]
        cortex_positions = get_positions(cache, 'Isocortex')  # these are in source order

        def get_source_index(position_3d):
            ind = np.where((cortex_positions == position_3d).all(axis=1))[0]
            if len(ind) == 0:
                raise Exception('Unknown voxel: {}'.format(position_3d))
            return ind[0]

        # Get indices of source positions
        indices = [get_source_index(p) for p in positions_3d]

        # Optionally omit experiments
        if args.omit_experiment_rank is not None:
            # sum cortex_weights at indices to see how impactful each experiment is for source voxels at positions_3d
            cortex_weights_for_selected_source_voxels = cortex_weights[indices]
            sums = np.sum(cortex_weights_for_selected_source_voxels, axis=0)
            ranked_experiment_indices = np.flip(np.argsort(sums))

            cortex_weights, nodes = remove_experiment(cortex_weights, nodes, ranked_experiment_indices[omit_experiment_rank])

        def get_mixture_for_target_voxel(target_index):
            target_weights = np.dot(nodes[:,target_index].T, cortex_weights.T)
            inclusion_threshold = 0.01 * np.max(target_weights)
            mixture = GaussianMixture2D()
            for g, p, ind in zip(gaussians, positions_3d, indices):
                weight = target_weights[ind]
                if weight > inclusion_threshold:
                    mixture.add(Gaussian2D(weight*g.weight, g.mean, g.covariance))

            return mixture

        result = []

        for i, target_index in enumerate(right_target_cortex_indices):
            if i % 100 == 0:
                print('{} of {}'.format(i, len(right_target_cortex_indices)))

            mixture = get_mixture_for_target_voxel(target_index)

            result.append(mixture.approx())

        target_positions = cortex_positions[right_target_cortex_indices]
        
        # FINDING MIXTURES IN CORTICAL COLUMNS
        # GOAL: For each propagated voxel, compute a weighted mixing with other voxels within a defined
        #       distance of the streamline

        # Matrix of standard deviations of the interlaminar connections from the CNN Mousenet paper
        il_stds = np.array([
            [142.5,    31.67,   63.33],
            [139.33,   114,     88.67],
            [95,       63.33,   133]
        ])

        # Find the average standard deviation to use as the std_dev parameter of the Gaussian weighting scheme
        avg_std = np.mean(il_stds)

        def gaussian_weighting(
            dist,
            std_dev=avg_std
        ):
            A = 1 / (std_dev * np.sqrt(2 * np.pi))
            base = np.exp(
                -0.5 * np.square(dist / std_dev)
            )

            return A * base

        min_std_dist = 3 # Number of standard deviations of distance that is considered in the weighting
        results_mixed = [] # List of propagated and mixed voxels for this area

        from streamlines import CompositeInterpolator
        with open('interpolator.pkl', 'rb') as file:
            ci = pickle.load(file)

        for voxel in target_positions:

            streamline = surface_to_surface_streamline(ci,voxel) # Streamline for voxel
            voxel_mixture = GaussianMixture2D() # Instantiate Mixture object
            
            # Find shortest distance between each propagated voxel and the streamline
            dists = shortest_distance(target_positions,streamline)

            # Iterate through all of the other propagated voxels of this area
            for gaussian, dist in zip(result, dists):
                       
                # If the distant voxel is within 3 standard devs (~290 microns) of the streamline, add it to the mixture
                if dist * 100 <= min_std_dist * avg_std:
                    weight = gaussian_weighting(dist * 100)
                    gaussian.weight = weight
                    voxel_mixture.add(gaussian)

            # Find mixture approximation for the voxel and add it to the results list for this area
            results_mixed.append(voxel_mixture.approx())

        # Dump propagated + mixed voxels for this area into a pickle file
        with open(f"propagated_and_mixed_{area}","wb") as file:
            pickle.dump(results_mixed, file)

if __name__ == "__main__":
    main()