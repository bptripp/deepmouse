import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.geodesic_flatmap import GeodesicFlatmap
from deepmouse.maps.map import get_positions
from deepmouse.topography import Gaussian2D


cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()

source_areas = ['VISp', 'AUDp', 'SSp-bfd', 'SSP-ul', 'SSp-m']
test_areas = ['VISp', 'VISal', 'VISam', 'VISl', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'VISa', 'VISrl']
# test_areas = ['SSs', 'AUDd', 'AUDpo', 'AUDv', 'MOp', 'MOs', 'RSP', 'TEa', 'ACAd', 'ACAv']
# test_areas = ['VISpor']

propagated = []
for sa in source_areas:
    with open('propagated {}'.format(sa), 'rb') as file:
        propagated.append(pickle.load(file))

flatmap = GeodesicFlatmap()

singular_values = []
for test_area in test_areas:
    positions_3d = get_positions(cache, test_area)
    indices = [flatmap.get_voxel_index(p) for p in positions_3d]

    multisensory_coordinates = np.zeros((2*len(source_areas), len(indices)))
    for i, index in enumerate(indices):
        voxel_coords = []
        for p in propagated:
            m = p[index].mean
            if len(m) > 1: # fix this in topography
                voxel_coords.extend([0, 0])
            else:
                voxel_coords.extend(m[0])
        multisensory_coordinates[:,i] = voxel_coords

    u, s, vh = np.linalg.svd(multisensory_coordinates)
    print(s)
    singular_values.append(s)

singular_values = np.array(singular_values)
cumulative_fraction = np.cumsum(singular_values, axis=1).T / np.sum(singular_values, axis=1)
# plt.plot(singular_values.T)
plt.plot(range(1, len(cumulative_fraction)+1), cumulative_fraction)
plt.xlabel('Singular value')
plt.ylabel('Cumulative fraction')
plt.legend(test_areas)
plt.tight_layout()
plt.savefig('singular-values.png')
plt.show()

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