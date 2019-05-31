import numpy as np
import pickle
from mcmodels.core import Mask, VoxelModelCache


"""
Code for estimating density profiles of inter-area connections from voxel model
of mouse connectome (Knox et al. 2019).
"""


class VoxelModel():
    # we make a shared instance because the model's state doesn't change
    # but it takes several seconds to instantiate, so we only want to do it once
    _instance = None

    def __init__(self):
        cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
        self.source_mask = cache.get_source_mask()
        self.source_keys = self.source_mask.get_key(structure_ids=None)

        with open('voxel-connectivity-weights.pkl', 'rb') as file:
            self.weights = pickle.load(file)
        with open('voxel-connectivity-nodes.pkl', 'rb') as file:
            self.nodes = pickle.load(file)

        self.structure_tree = cache.get_structure_tree()

    def get_weights(self, source_name='VISp2/3', target_name='VISpm4'):
        pre_id = self.structure_tree.get_id_acronym_map()[source_name]
        post_id = self.structure_tree.get_id_acronym_map()[target_name]

        pre_indices = []
        post_indices = []
        for i in range(len(self.source_keys)):
            if self.structure_tree.structure_descends_from(self.source_keys[i], pre_id):
                pre_indices.append(i)
            if self.structure_tree.structure_descends_from(self.source_keys[i], post_id):
                post_indices.append(i)

        weights_by_target_voxel = []
        for pi in post_indices:
            w = np.dot(self.weights[pre_indices,:], self.nodes[:,pi])
            weights_by_target_voxel.append(w)
        return weights_by_target_voxel

    @staticmethod
    def get_instance():
        """
        :return: Shared instance of VoxelModel
        """
        if VoxelModel._instance is None:
            VoxelModel._instance = VoxelModel()
        return VoxelModel._instance


class Target():
    def __init__(self, area, layer, external_in_degree=1000, voxel_model=None):
        """
        :param area: name of area
        :param layer: name of layer
        :param external_in_degree: Total neurons providing feedforward input to average
            neuron from other cortical areas.
        """
        self.target_name = area + layer
        self.e = external_in_degree

        self.voxel_model = VoxelModel.get_instance()

    def _get_voxels(self):
        """
        :return: List of voxels (coordinates) of target area/layer
        """
        pass

    def _get_external_sources(self):
        """
        :return: Names of sources (area, layer) that may project to this target,
            excluding other layers in the same area
        """
        pass

    def _get_source_voxels(self, target_voxel, source):
        pass

    def _get_mean_total_weight(self, source):
        pass

    def _get_gamma(self):
        pass

    def kernel_width_degrees(self, source, cortical_magnification):
        """
        :param source:
        :param cortical_magnification: mm cortex per degree visual angle
        :return: width (sigma) of Gaussian kernel approximation in degrees visual angle
        """
        return self.kernel_width_mm(source) / cortical_magnification

    def kernel_width_mm(self, source):
        sigmas = []
        for target_voxel in self._get_voxels():
            source_voxels = self._get_source_voxels(target_voxel, source)
            flatmap = get_flatmap(source_voxels)

            if not is_multimodal(flatmap):
                sigmas.append(find_radius(flatmap))

        return np.mean(sigmas)


def get_flatmap(souce_voxels):
    #TODO: implement
    return None

def is_multimodal(flatmap):
    """
    :param flatmap: flatmap of source voxel weights
    :return: True if weights have multiple dense regions, False if single dense region
    """
    #TODO: implement
    return False

def find_radius(flatmap):
    #TODO: deconvolve from model blur and flatmap blur
    #TODO: implement
    return .5


if __name__ == '__main__':
    print('starting')
    # vm = VoxelModel()
    # print('got voxel model')
    # weights = vm.get_weights(source_name='VISp2/3', target_name='VISpm4')
    # print('got weights')


    t = Target('VISpm', '4')
    print('foo')
    t = Target('VISpm', '4')
    print('foo')



