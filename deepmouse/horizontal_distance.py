import pickle
import numpy as np
import matplotlib.pyplot as plt
from mcmodels.core import VoxelModelCache
# from deepmouse.maps.util import get_default_structure_tree
# from deepmouse.maps.map import get_positions
# from deepmouse.streamlines import get_streamline, CompositeInterpolator
from maps.util import get_default_structure_tree
from maps.map import get_positions
from streamlines import get_streamline, CompositeInterpolator

def line_segment_distances(points, a, b):
    # https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
    points = np.atleast_2d(points)
    d = np.divide(b - a, np.linalg.norm(b - a))
    s = np.dot(a - points, d)
    t = np.dot(points - b, d)
    h = np.maximum.reduce([s, t, np.zeros(len(points))])
    c = np.linalg.norm(np.cross(points - a, d), axis=1)
    return np.hypot(h, c)


def shortest_distance(positions, streamline):
    # loop through line segments in streamline, keep track of shortest distance to each point
    result = np.full(positions.shape[0], np.inf)

    def distances_from_segment(start_index, len):
        a = streamline[start_index, :]
        b = streamline[start_index + len, :]
        return line_segment_distances(positions, a, b)

    for i in range(0, streamline.shape[0]-2, 2):
        distances = distances_from_segment(i, 2)
        result = np.minimum(result, distances)

    # check end too in case last segment is one from end
    distances = distances_from_segment(streamline.shape[0]-3, 2)
    result = np.minimum(result, distances)

    return result


def laterally_close(position, cutoff=4):
    s_both = surface_to_surface_streamline(ci, position)
    shortest = shortest_distance(positions_all, s_both)
    close_indices = np.where(shortest < cutoff)
    close_distances = shortest[close_indices]
    return close_indices, close_distances

cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
structure_tree = get_default_structure_tree()

with open('interpolator.pkl', 'rb') as file:
    ci = pickle.load(file)


def surface_to_surface_streamline(ci, position):
    up = get_streamline(ci, position)
    down = get_streamline(ci, position, to_surface=False)
    return np.concatenate((np.flip(down, axis=0), up[1:, :]))

def main():
    position = [69, 10, 61]
    s = get_streamline(ci, position)
    s_down = get_streamline(ci, position, to_surface=False)
    s_both = surface_to_surface_streamline(ci, position)

    # plt.figure(figsize=(8,3))
    # plt.subplot(121)
    # plt.plot(s[:,0], s[:,1])
    # plt.plot(s_down[:,0], s_down[:,1])
    # plt.plot(s_both[:,0], s_both[:,1], 'k--')
    # plt.subplot(122)
    # plt.plot(s[:,0], s[:,2])
    # plt.plot(s_down[:,0], s_down[:,2])
    # plt.plot(s_both[:,0], s_both[:,2], 'k.')
    # plt.show()

    positions_all = get_positions(cache, 'Isocortex') # these are in source order
    # import time
    # start = time.time()
    # shortest = shortest_distance(positions_all, s_both)
    # print(time.time() - start)
    # plt.hist(shortest, 100)
    # plt.show()

    test_voxel = [69, 10, 61]
    ind, dist = laterally_close(test_voxel)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(positions_all[ind,0], positions_all[ind,1], positions_all[ind,2], c=dist)
    ax.scatter(test_voxel[0], test_voxel[1], test_voxel[2], marker='x', s=80)
    ax.figure.savefig("test2.png")
    # plt.show()
