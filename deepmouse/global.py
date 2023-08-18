import pickle
import numpy as np
import matplotlib.pyplot as plt
from deepmouse.maps.map import right_target_indices
from deepmouse.maps.util import get_area_positions, get_default_structure_tree, get_id
from deepmouse.maps.flatmap import FlatMap
from deepmouse.horizontal_distance import surface_to_surface_streamline, shortest_distance
from deepmouse.streamlines import CompositeInterpolator
from deepmouse.topography import load_weights
from mcmodels.core import VoxelModelCache

# TODO: are coords correlated in low-dimensional areas? (magnitude)
# TODO: check ovals use CoM calculation
# TODO: column weights should be weighted sum or weighted average over voxels?

cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')

with open('interpolator.pkl', 'rb') as file:
    ci = pickle.load(file)

weights, nodes = load_weights('data_files/')  # weights 226346 x 428


class Columns:
    def __init__(self, area, sigma=.5):
        self.area = area
        self.flatmap = FlatMap.get_instance(area)
        self.centres = self.flatmap.get_column_centres()

        self.streamlines = []
        for centre in self.centres.T:
            streamline_origin = self.flatmap.get_position_3d(centre)
            self.streamlines.append(surface_to_surface_streamline(ci, streamline_origin))

        self.positions_3d = get_area_positions(cache, self.area)

        self.distances = []
        for streamline in self.streamlines:
            d = shortest_distance(self.positions_3d.T, streamline)
            self.distances.append(d)
        self.distances = np.array(self.distances)
        self.mixing_weights = np.exp(-self.distances**2 / 2 / sigma**2)

    def get_mesoscale_column_weights(self):
        """
        :return: A matrix similar to the weights matrix of the mesoscale model, except that it contains one weight per
            cortical column rather than one weight per voxel, and it only includes weights for this object's area
        """
        area_weights = get_mesoscale_weights_for_area(self.area)
        return np.matmul(self.mixing_weights, area_weights)


def get_mesoscale_weights_for_area(area):
    structure_tree = get_default_structure_tree()
    structure_id = get_id(structure_tree, area)

    source_mask = cache.get_source_mask()
    source_keys = source_mask.get_key(structure_ids=None)  # structure ids for all voxels

    source_area_indices = []
    for i in range(len(source_keys)):
        if structure_tree.structure_descends_from(source_keys[i], structure_id):
            source_area_indices.append(i)

    return weights[source_area_indices, :]


def get_mesoscale_nodes_for_area(area):
    structure_tree = get_default_structure_tree()
    structure_id = get_id(structure_tree, area)

    target_mask = cache.get_target_mask()
    target_keys = target_mask.get_key(structure_ids=None) # structure ids for all voxels

    target_area_indices = []
    for i in range(len(target_keys)):
        if structure_tree.structure_descends_from(target_keys[i], structure_id):
            target_area_indices.append(i)

    target_area_indices = np.array(target_area_indices)
    r = right_target_indices(cache, area=area)
    right_target_area_indices = target_area_indices[r]

    return nodes[:, right_target_area_indices]


def mean_anterograde_connection_strengths(weights, nodes):
    """
    :param weights: column-wise weights; result of Columns.get_mesoscale_column_weights()
    :param nodes: mesoscale model nodes
    :return: mean outbound connection strength of each column
    """
    connection_strengths = np.matmul(weights, nodes)
    return np.mean(connection_strengths, axis=1)


def uniformity_score(connection_weights):
    """
    :param connection_weights: connection weights into a single target voxel
    :return: a metric of uniformity of input (0 to 1)
    """
    s = np.sort(connection_weights)
    peak = np.mean(s[-3:])
    # normalized = np.minimum(1, (connection_weights / peak)**.5)
    normalized = np.minimum(1, connection_weights / peak)
    return np.mean(normalized)


def mean_uniformity_score(connection_weights):
    """
    :param connection_weights: matrix of connection weights (each column for a different target voxel)
    :return: weighted mean metric across target voxels
    """
    scores = []
    strengths = []
    for i in range(connection_weights.shape[1]):
        scores.append(uniformity_score(connection_weights[:, i]))
        strengths.append(np.mean(connection_weights[:,i]))
    return np.dot(scores, strengths) / np.sum(strengths)


def area_area_weight(source_area, target_area):
    w = get_mesoscale_weights_for_area(source_area)
    n = get_mesoscale_nodes_for_area(target_area)
    voxel_voxel_weights = np.matmul(w, n)
    return np.sum(voxel_voxel_weights.flatten())


if __name__ == '__main__':
    columns = Columns('VISp')
    VISp_column_weights = columns.get_mesoscale_column_weights()
    VISl_nodes = get_mesoscale_nodes_for_area('VISl')
    VISp_VISl_connection_strengths = np.matmul(VISp_column_weights, VISl_nodes) # shape #columns x #voxels
    print(uniformity_score(VISp_VISl_connection_strengths[:, 0]))
    print(mean_uniformity_score(VISp_VISl_connection_strengths))

    plt.hist(mean_anterograde_connection_strengths(VISp_column_weights, nodes))
    plt.title('Mean anterograde connection strengths of VISp columns')
    plt.show()

    example_voxel = 200
    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    plt.scatter(columns.centres[0,:], columns.centres[1,:], c=VISp_VISl_connection_strengths[:, example_voxel])
    plt.colorbar()
    plt.axis('equal')
    plt.subplot(1,2,2)
    s = np.sort(VISp_VISl_connection_strengths[:, example_voxel])
    peak = np.mean(s[-3:])
    compressed = np.minimum(1, (VISp_VISl_connection_strengths[:, example_voxel] / peak) ** .5)
    print('score: {}'.format(uniformity_score(VISp_VISl_connection_strengths[:, example_voxel])))
    plt.scatter(columns.centres[0,:], columns.centres[1,:], c=compressed)
    plt.colorbar()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('example_mesoscale_connection_density.png')
    plt.show()


    with open('uniformity_results.pkl', 'rb') as file:
        foo = pickle.load(file)
        print(foo)

    volumes = []
    for area in foo['source_areas']:
        p = get_area_positions(cache, area)
        volumes.append(p.shape[1])

    # mu = np.mean(foo['uniformities'], axis=1)
    # plt.scatter(volumes, mu)
    # plt.xlabel('Volume (voxels)')
    # plt.ylabel('Mean outbound uniformity')
    # plt.savefig('uniformity-vs-volume.png')
    # plt.show()

    plt.figure(figsize=(14,6))
    plt.subplot(121)
    # plt.imshow(foo['uniformities'])
    u = foo['uniformities'].copy()
    for i in range(u.shape[0]):
        u[i,:] = u[i,:] / np.mean(u[i,:])
    plt.imshow(u)
    plt.xticks(range(len(foo['source_areas'])), foo['source_areas'], rotation=90)
    plt.yticks(range(len(foo['source_areas'])), foo['source_areas'])
    plt.colorbar()
    plt.subplot(122)
    w = foo['area_area_weights'].copy()
    for i in range(w.shape[0]):
        w[i,:] = w[i,:] / np.sum(w[i,:])
    plt.imshow(np.log(w))
    plt.xticks(range(len(foo['source_areas'])), foo['source_areas'], rotation=90)
    plt.yticks(range(len(foo['source_areas'])), foo['source_areas'])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('uniformities_normalized.png')
    plt.show()

    plt.figure(figsize=(6,6))
    a = np.log(w)
    a = (a - np.min(a.flatten())) / (np.max(a.flatten()) - np.min(a.flatten()))
    print(a.shape)
    print(u.shape)
    plt.imshow(foo['uniformities'], alpha=a)
    plt.xticks(range(len(foo['source_areas'])), foo['source_areas'], rotation=90)
    plt.yticks(range(len(foo['source_areas'])), foo['source_areas'])
    # plt.colorbar()
    plt.savefig('uniformities_alpha.png')
    plt.show()


    areas = ['VISpm', 'VISli', 'VISpor', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl', 'AId',
             'AIp', 'AIv', 'RSPagl', 'RSPv', 'VISa', 'VISrl', 'TEa', 'PERI', 'ECT',
             'MOp', 'MOs', 'SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr', 'SSp-un', 'SSs', 'GU',
             'VISC', 'AUDd', 'AUDp', 'AUDpo', 'AUDv', 'VISal', 'VISam', 'VISl', 'VISp',
    ]
    # TODO: code not working for some areas: 'FRP', 'VISrll', 'VISlla', 'VISmma','VISmmp', 'ORBv', 'VISm', 'RSPd', 'PTLp', 'VISpl'

    source_areas = areas
    target_areas = areas
    uniformities = np.zeros((len(source_areas), len(target_areas)))
    area_area_weights = np.zeros((len(source_areas), len(target_areas)))
    for i, source_area in enumerate(source_areas):
        print('Source: {}'.format(source_area))
        columns = Columns(source_area)
        column_weights = columns.get_mesoscale_column_weights()
        for j, target_area in enumerate(target_areas):
            print(' Target: {}'.format(target_area))
            VISl_nodes = get_mesoscale_nodes_for_area(target_area)
            connection_strengths = np.matmul(column_weights, VISl_nodes)
            mus = mean_uniformity_score(connection_strengths)
            aaw = area_area_weight(source_area, target_area)
            uniformities[i,j] = mus
            area_area_weights[i,j] = aaw
            print('  Uniformity: {}'.format(mus))
            print('  Area-area weight: {}'.format(aaw))

    with open('uniformity_results.pkl', 'wb') as file:
        pickle.dump({
            'source_areas': source_areas,
            'target_areas': target_areas,
            'uniformities': uniformities,
            'area_area_weights': area_area_weights
        }, file)

