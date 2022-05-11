"""
Simplified kernel-based model of voxel-based model.
"""
import pickle
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mcmodels.core import Mask, VoxelModelCache
from deepmouse.maps.util import search_id_by_acronym
from deepmouse.maps.flatmap import FlatMap
from deepmouse.garrett import VoxelRetinotopy


print('starting')
cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
# source_mask = Mask.from_cache(cache, hemisphere_id=2, structure_ids=[315]) # 315 cortex; arg affects key length; what was 669?
source_mask = cache.get_source_mask()
target_mask = cache.get_target_mask()
# cache.get_voxel_connectivity_array()

source_keys = source_mask.get_key(structure_ids=None)
source_key_volume = source_mask.map_masked_to_annotation(source_keys)

target_keys = target_mask.get_key(structure_ids=None)
target_key_volume = target_mask.map_masked_to_annotation(target_keys)

# print(len(source_keys))
# print(source_key_volume.shape)
# print(len(target_keys))
# print(target_key_volume.shape)
# print(np.sum(target_key_volume > 0))

print('tree')

structure_tree = cache.get_structure_tree()
pre_id = structure_tree.get_id_acronym_map()['VISp2/3']
# pre_id = structure_tree.get_id_acronym_map()['VISp5']
post_id = structure_tree.get_id_acronym_map()['VISpm4']

mask_indices = np.array(source_mask.mask.nonzero())

pre_indices = []
pre_positions = []
post_indices = []
for i in range(len(source_keys)): # only one hemisphere for now
    if structure_tree.structure_descends_from(source_keys[i], pre_id):
        pre_indices.append(i)
        pre_positions.append(mask_indices[:,i])
    if structure_tree.structure_descends_from(source_keys[i], post_id):
        post_indices.append(i)

print('indices')

foo_indices = []
for i in range(len(target_keys)):
    if structure_tree.structure_descends_from(target_keys[i], post_id):
        foo_indices.append(i)

# print(len(post_indices))
# print(len(foo_indices))
# print(len(source_keys))
# print(len(target_keys))
# print(target_keys)
# assert False


# print(len(pre_indices))
# print(len(pre_positions))
# print(len(post_indices))
# print(post_indices)


# print(len(mask_indices[0])) # looks like idx is 3 by length of source_keys
# so I can find position of pi=post_indices[x] as [idx[0][pi], idx[1][pi], idx[2][pi]]?

print('opening voxel model')

with open('voxel-connectivity-weights.pkl', 'rb') as file:
    weights = pickle.load(file)
with open('voxel-connectivity-nodes.pkl', 'rb') as file:
    nodes = pickle.load(file)
# print(weights.shape)
# print(nodes.shape)
# print(np.mean(nodes))
# print(np.mean(nodes[:,:226346]))
# print(np.mean(nodes[:,226346:]))

print('flatmap')

flatmap = FlatMap()
flatmap._fit()
# positions = np.array(source_mask.mask.nonzero())


def flatmap_weights(indices, weights):
    rel_weights = weights / max(weights)
    for i in range(len(indices)):
        # index = indices[i]
        # position = positions[:,i]
        position2d = flatmap.get_position_2d(pre_positions[i])
        color = [rel_weights[i], 0, 1-rel_weights[i], .5]
        # print(rel_weights[i])
        # if rel_weights[i] > .001:
        plt.scatter(position2d[0], position2d[1], c=color)
    plt.xticks([]), plt.yticks([])

# vr = VoxelRetinotopy()

# #TODO: a few of these are nan
# def visual_space_weights(indices, weights):
#     rel_weights = weights / max(weights)
#     coords = []
#     weighted_sum = np.zeros(2, dtype=float)
#     w = []
#     for i in range(len(indices)):
#         c = vr.get_retinal_coords(pre_positions[i])
#         if c is None or np.isnan(c[0]) or np.isnan(c[1]):
#             print('position unknown')
#         else:
#             if rel_weights[i] > .05: # avoid pulling toward centre of V1 due to diffuse weight
#                 coords.append(c)
#                 weighted_sum = weighted_sum + np.array(c) * rel_weights[i]
#                 w.append(rel_weights[i])
#     centroid = weighted_sum / sum(w)
#     for i in range(len(coords)):
#         # position2d = flatmap.get_position_2d(pre_positions[i])
#         color = [w[i], 0, 1-w[i], .5]
#         # if coords is None:
#         #     print('position unknown')
#         # else:
#         plt.scatter(coords[i][0]-centroid[0], coords[i][1]-centroid[1], c=color)
#     plt.xlim((-60, 60))
#     plt.ylim((-40, 40))
#     plt.xticks([]), plt.yticks([])

# plt.imshow(weights[pre_indices,:])
# plt.plot(nodes[:,post_indices[31]])
# plt.show()

plt.figure()
print(len(post_indices))
for i in range(len(post_indices)):
    print(i)
    pi = post_indices[i]
    input_weights = np.dot(weights[pre_indices,:], nodes[:,pi])
    plt.subplot(7,7,i+1)
    flatmap_weights(pre_indices, input_weights)
    # visual_space_weights(pre_indices, input_weights)
plt.show()
# print(input_weights.shape)
# print(pre_indices)
# print('min: {} max: {}'.format(min(input_weights), max(input_weights)))


# def centroid(positions, weights):
#     numerator = 0
#     denominator = 0
#     for i in range(len(weights)):
#         print(positions[i])
#         print(weights[i])
#         numerator += positions[i] * weights[i]
#         denominator += weights[i]
#     return numerator / denominator
#
# def distances(positions, centroid):
#     result = []
#     for position in positions:
#         difference = np.array(position) - np.array(centroid)
#         result.append(np.linalg.norm(difference))
#     return result

# # find inputs to one of the post voxels
# # hopefully hemisphere accounted for properly above
# for pi in post_indices:
#     input_weights = np.dot(weights[pre_indices,:], nodes[:,pi])
#
#     c = centroid(pre_positions, input_weights)
#     d = distances(pre_positions, c)
#
#     rel_input_weights = np.array(input_weights) / max(input_weights)
#     plt.scatter(d, rel_input_weights)
#
# plt.show()


# start = timer()
# foo = []
# # x = 315
# x = search_id_by_acronym(structure_tree, 'VISp4')[0]
# for k in key:
#     foo.append(structure_tree.structure_descends_from(k, x))
# print('time: {}'.format(timer() - start))
# plt.plot(foo)
# plt.show()
#


# structure_tree = cache.get_structure_tree()
# # areas = ('VISp', 'VISl', 'VISal', 'VISlla', 'VISrl', 'VISa', 'VISm', 'VISam', 'VISpm', 'VISli')
# # VISlm missing from structure tree
# # no voxels for VISrl and VISa
# # Stefan diagrammed VISa and VISp (in the posterior sense of Zhuang et al. I think)
# #  VISa is there but no voxels; VISpl seems to be part of Zhuang et al. VISp
# # areas = ('VISp', 'VISlm', 'VISrl', 'VISpm', 'VISli', 'VISal', 'VISam', 'VISpor')
# # areas = ('VISp', 'VISpm', 'VISam', 'VISpor') # simplified network
# areas = ('VISp', 'VISpm')
# for area in areas:
#     id = search_id_by_acronym(structure_tree, area + '4')[0]
#     print(id)
# # l4_ids = [search_id_by_acronym(structure_tree, area+'5')[0] for area in areas]
#

