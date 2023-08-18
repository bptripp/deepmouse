import pickle
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.geodesic_flatmap import GeodesicFlatmap
from deepmouse.maps.map import get_positions
from deepmouse.topography import Gaussian2D


cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()

# source_areas = ['SSp-n']
source_areas = ['VISp', 'AUDp', 'SSp-bfd', 'SSP-ul']
# test_areas = ['VISp', 'VISal', 'VISam', 'VISl', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'VISa', 'VISrl']
# test_areas = ['SSs', 'AUDd', 'AUDpo', 'AUDv', 'MOp', 'MOs', 'RSP', 'TEa', 'ACAd', 'ACAv']
test_areas = ['RSP']

propagated = []
for sa in source_areas:
    with open('generated/propagated {}'.format(sa), 'rb') as file:
        propagated.append(pickle.load(file))

flatmap = GeodesicFlatmap()


def plot_detail(ml_coordinates, ap_coordinates, multisensory_weights, p):
    norm_weights = multisensory_weights / max(multisensory_weights.flatten())

    fig = plt.figure(figsize=(6.5,2.7))
    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap('jet'))

    def scatter(coords, weights):
        ax.scatter(p[:,0], p[:,1], p[:,2],
                   marker='.',
                   c=scalar_map.to_rgba(coords),
                   alpha=weights)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.zaxis.set_ticks([])

    for i in range(4):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        plt.title(source_areas[i])
        scatter(ml_coordinates[i,:], norm_weights[i,:])

        ax = fig.add_subplot(2, 4, i+5, projection='3d')
        scatter(ap_coordinates[i,:], norm_weights[i,:])

    plt.tight_layout()
    plt.savefig('generated/RSP-mix.png')
    plt.show()



# singular_values = []
# for test_area in test_areas:
#     positions_3d = get_positions(cache, test_area)
#     indices = [flatmap.get_voxel_index(p) for p in positions_3d]
#
#     ml_coordinates = np.zeros((len(source_areas), len(indices)))
#     ap_coordinates = np.zeros((len(source_areas), len(indices)))
#     multisensory_weights = np.zeros((len(source_areas), len(indices)))
#     for i, index in enumerate(indices):
#         for j, p in enumerate(propagated):
#             m = p[index].mean
#             ml_coordinates[j,i] = m[0]
#             ap_coordinates[j,i] = m[1]
#             multisensory_weights[j,i] = p[index].weight
#
#     plot_detail(ml_coordinates, ap_coordinates, multisensory_weights, positions_3d)
#
#     scaled_ml_coordinates = np.multiply(ml_coordinates, multisensory_weights)
#     scaled_ap_coordinates = np.multiply(ap_coordinates, multisensory_weights)
#     scaled_coordinates = np.concatenate((scaled_ml_coordinates, scaled_ap_coordinates), axis=0)
#     u, s, vh = np.linalg.svd(scaled_coordinates)
#     print(s)
#     singular_values.append(s)
#
# plt.figure(figsize=(3,2.5))
# singular_values = np.array(singular_values)
# cumulative_fraction = np.cumsum(singular_values, axis=1).T / np.sum(singular_values, axis=1)
# plt.plot(range(1, len(cumulative_fraction)+1), cumulative_fraction)
# plt.xlabel('Singular value')
# plt.ylabel('Cumulative fraction')
# plt.ylim([0, 1])
# plt.legend(test_areas)
# plt.tight_layout()
# # plt.savefig('singular-values.png')
# plt.show()

# R = np.corrcoef(multisensory_coordinates)
#
# labels = []
# for source_area in source_areas:
#     labels.append('{}-ml'.format(source_area))
#     labels.append('{}-ap'.format(source_area))
#
# sns.heatmap(R, xticklabels=labels, yticklabels=labels)
# plt.tight_layout()
# plt.savefig('VIRpor-correlations.png')
# plt.show()