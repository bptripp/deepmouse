import numpy as np
from deepmouse.maps.util import get_voxel_model_cache, get_positions, get_id, get_default_structure_tree
from deepmouse.geodesic_flatmap import GeodesicFlatmap
from deepmouse.maps.flatmap import FlatMap
"""
This code assigns coordinates to each voxel of each primary sensory area based on its flatmap
position. It propagates these coordinates through the connectivity model. This produces, for each
non-primary voxel, an estimate of its coordinates in sensory space.
"""

# cache = get_voxel_model_cache()
# structure_tree = get_default_structure_tree()
#
# area = 'SSp-ul'
# id = get_id(structure_tree, area)
# print("'{}': {}".format(area, id))
# positions = get_positions(cache, id)
# print(len(positions))

# area = 'VISp'
# area = 'AUDp'
# area = 'PIR'
# area = 'SSp-bfd'
# area = 'SSp-m'
# area = 'SSp-n'
area = 'SSp-ul'
flatmap = FlatMap(area=area)
flatmap._fit()
# print(flatmap.centre)
# print(flatmap.radius)
# print(np.mean(flatmap.positions_3d.T, axis=0))
# print(np.mean(flatmap.positions_3d.T, axis=0) - flatmap.centre)
# flatmap._plot_residuals()
# positions_2d = []
# for position_3d in flatmap.positions_3d.T:
#     position_2d = flatmap.get_position_2d(position_3d)
#     positions_2d.append(position_2d)
# positions_2d = np.array(positions_2d)
positions_2d = flatmap.positions_2d.T

centre = np.mean(positions_2d, axis=0)
squared_distances = np.sum((positions_2d - centre)**2, axis=1)
sd = (np.sum(squared_distances)/len(positions_2d))**.5
print(centre)
print(sd)

rel_positions = positions_2d - centre
rel_positions = rel_positions / sd

import matplotlib.pyplot as plt
plt.scatter(rel_positions[:,0], rel_positions[:,1])
# plt.scatter(positions_2d[:,0], positions_2d[:,1])
plt.axis('equal')
plt.show()

# id = get_id(structure_tree, area)
# print("'{}': {}".format(area, id))
# positions = get_positions(cache, id)


# flatmap = GeodesicFlatmap(area=area)
#
# surface_positions = []
# positions_list = positions.tolist()
# for i, vertex in enumerate(flatmap.vertices):
#     vertex = vertex.astype('int32').tolist()
#     if vertex in positions_list:
#         surface_positions.append([flatmap.ml_position[i], flatmap.ap_position[i]])
# surface_positions = np.array(surface_positions)
#
# centre = np.mean(surface_positions, axis=0)
# squared_distances = np.sum((surface_positions - centre)**2, axis=1)
# sd = (np.sum(squared_distances)/len(surface_positions))**.5
#
# print(centre)
# print(sd)
#
# rel_positions = surface_positions - centre
# rel_positions = rel_positions / sd
#
# import matplotlib.pyplot as plt
# plt.scatter(rel_positions[:,0], rel_positions[:,1])
# plt.show()
