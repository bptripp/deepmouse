"""
This code calculates streamlines from white matter to pia, for the purpose of
projecting features onto the cortical surface. The method is described in
Harris et al. (2019), Nature. It is similar to Jones et al. (2000) Human Brain Mapping.
"""

import pickle
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssla
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt #TODO: plot voxels
from mcmodels.core import VoxelModelCache
from deepmouse.maps.map import get_positions
from deepmouse.maps.util import get_child_ids, get_default_structure_tree, print_descriptions
from deepmouse.maps.util import get_name, get_acronym, get_id

cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
structure_tree = get_default_structure_tree()


def is_layer(acronym):
    result = False
    # note '2' appears in PL and ILA
    for ending in ('1', '2', '2/3', '4', '5', '6a', '6b'):
        if acronym.endswith(ending):
            result = True
            break
    return result


def get_child_area_ids(structure_id):
    result = []
    for id in get_child_ids(structure_tree, structure_id):
        if not is_layer(get_acronym(structure_tree, id)):
            result.append(id)
    return result


def get_leaf_areas(root_id=315): # 315 is isocortex
    result = []

    for id in get_child_ids(structure_tree, root_id):
        if not is_layer(get_acronym(structure_tree, id)):
            x = get_leaf_areas(root_id=id)
            result.extend(x)
            if len(x) == 0: # id is a leaf area
                result.append(id)

    """
    The list produced by the above code has discrepancies with Brain Explorer 2:     
     - Missing 894 (RSPagl)
     - Has 480149202 (VISrll) in place of 329 (SSp-bfd)
     - Has 480149230 (VISlla) instead of 1011 (AUDd)
     - Extra areas ORBv, VISmma, VISmmp, VISm
     
     Relatedly, 
     - ORBv doesn't have layers as children.
     - VISrll, VISlla, VISmma, VISmmp, VISm layers don't have voxels 
     
    To avoid missing anything we manually add 894, 329, and 1011.
    """
    if root_id == 315:
        result.extend([894, 329, 1011])

    return result


def get_acronyms_for_layer(target_layer):
    """
    :param target_layer: in ('1', '2/3', '4', '5', '6a', '6b')
    :return: list of structure acronyms of given layer, e.g. 'MO1', etc.
    """

    # Note most top-level areas have layers as children.
    # AUD, AI, RSP don't (they have layers as grandchildren).
    # However, the ones that do don't have voxels assigned to their layers.
    # So we use get_leaf_areas.

    layer_acronyms = []
    for area_id in get_leaf_areas():
        found_layer = False
        for layer_id in get_child_ids(structure_tree, area_id):
                acronym = get_acronym(structure_tree, layer_id)
                if acronym.endswith(target_layer):
                    layer_acronyms.append(acronym)
                    found_layer = True
                    break
        if not found_layer:
            area = get_acronym(structure_tree, area_id)
            print('Layer {} not found for {}'.format(target_layer, area))

    return layer_acronyms


def get_positions_for_layer(target_layer):
    positions = []
    for acronym in get_acronyms_for_layer(target_layer):
        p = get_positions(cache, acronym)
        print('{}: {}'.format(acronym, p.shape))
        positions.extend(p)
    positions = np.array(positions)
    print('total')
    print(positions.shape)
    return(positions)


def save_positions_per_layer():
    positions = {}
    layers = ('1', '2/3', '4', '5', '6a', '6b')
    for layer in layers:
        positions[layer] = get_positions_for_layer(layer)

    with open('positions-per-layer.pkl', 'wb') as file:
        pickle.dump(positions, file)


def load_positions_per_layer():
    with open('positions-per-layer.pkl', 'rb') as file:
        positions = pickle.load(file)

    return positions


def get_neighbours(position):
    neighbours = np.array([position]*6)

    neighbours[0,0] = neighbours[0,0] - 1
    neighbours[1,0] = neighbours[1,0] + 1
    neighbours[2,1] = neighbours[2,1] - 1
    neighbours[3,1] = neighbours[3,1] + 1
    neighbours[4,2] = neighbours[4,2] - 1
    neighbours[5,2] = neighbours[5,2] + 1

    return neighbours


def on_edge(position, positions):
    """
    Checks whether a voxel is at the edge of a volume. Example: to check whether an
    L6 voxel is at the edge of the cortex (vs. either internal to L6 or bordering L5),
    pass the union of L5 and L6 voxels as the positions arg. If an L6 voxel is at the
    edge of the L5/L6 volume, it is at the edge of the cortex.

    :param position: a voxel position to check
    :param positions: all voxel positions that make up a certain volume
    :return: True if positions does not contain all the neighbours of position
    """
    neighbours = get_neighbours(position)

    for neighbour in neighbours:
        if not (positions == neighbour).all(axis=1).any():
            return True

    return False


def get_neighbour_indices(position, positions):
    # note interior points should have six neighbours
    neighbours = get_neighbours(position)

    result = []
    for neighbour in neighbours:
        index = np.where((positions == neighbour).all(axis=1))[0]
        # print(position)
        # print(positions)
        # print((positions == neighbour).all(axis=1))
        # print(index)
        if index:
            result.append(index[0])

    return result


def laplace_solution(positions, is_outer, is_inner):
    """
    Solves Laplace's equation to estimate depth of inner voxels from boundary conditions
    at inner and outer edge of cortex.

    :param positions: voxel positions of cortex
    :param is_outer: flag for each position; True if part of outer surface
    :param is_inner: flag for each position; True if part of inner surface
    :return: estimated depths of all voxels
    """
    n = positions.shape[0]
    A = ss.lil_matrix((n,n))
    b = np.zeros(n)

    # finite difference method
    for i, position in enumerate(positions):
        if i % 1000 == 0:
            print('Finite difference {} of {}'.format(i, n))

        # note step sizes in each direction are one, so no denominators needed
        if not (is_outer[i] or is_inner[i]):
            ni = get_neighbour_indices(position, positions)
            for index in np.sort(ni):
                A[i,index] = 1

            A[i,i] = -len(ni) # normally = -2*3, but we have some edges without boundary conditions

        if is_outer[i]:
            A[i,i] = 1
            b[i] = 0

        if is_inner[i]:
            A[i,i] = 1
            b[i] = 7

    solution = ssla.lsqr(A, b)
    return solution[0]

    print('Laplace solution mean: {} residual norm: {}'.format(np.mean(solution[0]), solution[3]))


def find_edge(candidates, volume):
    result = []

    n = len(candidates)
    for i, candidate in enumerate(candidates):
        if i % 100 == 0:
            print('Checking {} of {}'.format(i, n))
        if on_edge(candidate, volume):
            result.append(candidate)

    return np.array(result)


def save_inner_and_outer_edges():
    positions = load_positions_per_layer()

    positions_123 = np.concatenate((positions['1'], positions['2/3']))
    outer_edge = find_edge(positions['1'], positions_123)
    # print(positions['1'].shape)
    # print(outer_edge.shape)

    positions_6 = np.concatenate((positions['6a'], positions['6b']))
    positions_56 = np.concatenate((positions['5'], positions_6))
    inner_edge = find_edge(positions_6, positions_56)
    # print(positions_56.shape)
    # print(positions_6.shape)
    # print(inner_edge.shape)

    edges = {'inner': inner_edge, 'outer': outer_edge}

    with open('inner-outer-edges.pkl', 'wb') as file:
        pickle.dump(edges, file)


def load_inner_and_outer_edges():
    with open('inner-outer-edges.pkl', 'rb') as file:
        edges = pickle.load(file)

    return edges


def save_inner_outer_flags():
    positions = load_positions_per_layer()

    positions_all = np.concatenate((
        positions['1'],
        positions['2/3'],
        positions['4'],
        positions['5'],
        positions['6a'],
        positions['6b'],
    ))

    edges = load_inner_and_outer_edges()

    inner_edge = edges['inner']
    outer_edge = edges['outer']

    is_inner = np.full(positions_all.shape[0], False)
    is_outer = np.full(positions_all.shape[0], False)

    for i in range(positions_all.shape[0]):
        if i % 1000 == 0:
            print('Checking inner/outer edge {} of {}'.format(i, positions_all.shape[0]))
        p = positions_all[i,:]
        is_inner[i] = np.where((inner_edge == p).all(axis=1))[0].size
        is_outer[i] = np.where((outer_edge == p).all(axis=1))[0].size

    with open('inner-outer-flags.pkl', 'wb') as file:
        pickle.dump({'is_inner': is_inner, 'is_outer': is_outer}, file)


def load_inner_outer_flags():
    with open('inner-outer-flags.pkl', 'rb') as file:
        flags = pickle.load(file)

    return flags


def get_interpolator(positions, depths):
    """
    :param positions: voxel positions (n x 3)
    :param depths: cortical depth of each position
    :return: RBF interpolator
    """
    # return Rbf(positions[:,0], positions[:,1], positions[:,2], depths, function='gaussian', epsilon=1)
    return Rbf(positions[:,0], positions[:,1], positions[:,2], depths, epsilon=1) #smaller edge effects than gaussian


def get_gradient(interpolator, position):
    """
    :param interpolator: sub-voxel interpolator of depths (from estimated depths of voxels)
    :param position: position at which to estimate gradient
    :return: finite-difference approximation of gradient
    """
    d = .1 # fraction voxel to move for finite difference

    px, py, pz = position[0], position[1], position[2]
    gx = (interpolator(px + d, py, pz) - interpolator(px - d, py, pz)) / (2*d)
    gy = (interpolator(px, py + d, pz) - interpolator(px, py - d, pz)) / (2*d)
    gz = (interpolator(px, py, pz + d) - interpolator(px, py, pz - d)) / (2*d)

    return np.array([gx, gy, gz])


def get_streamline(interpolator, origin, step_size=0.1):
    """
    TODO: need surface? edges outside cortex?
    Gradient descent to outside of cortex.
    """
    p = origin
    depth = interpolator(p[0], p[1], p[2])

    if not depth:
        print(depth)
        print(origin)

    streamline = [p]
    c = 0
    while depth > 0.25: # unstable at the very edge
        c = c + 1
        if c > 500:
            print('stuck')
            print(streamline)
            break
        g = get_gradient(interpolator, p)
        step = - step_size * g / np.linalg.norm(g)
        p = p + step
        streamline.append(p)
        depth = interpolator(p[0], p[1], p[2])

    return np.array(streamline)


def get_slice_indices(positions, low_x, high_x):
    result = []
    for i, p in enumerate(positions):
        if low_x < p[0] < high_x:
            result.append(i)
    return result


def get_closest_voxel(position, edge_positions):
    differences = position - edge_positions
    distances = np.linalg.norm(differences, axis=1)
    result = edge_positions[np.argmin(distances),:]
    return result


class CompositeInterpolator:
    """
    Wrapper for multiple interpolators for different slices of the voxel space. We need
    this because it's too slow and memory intensive to build one large interpolator.
    """
    def __init__(self, positions, depths):
        n = 10 # number of slices
        buffer = 2
        self.boundaries = np.linspace(min(positions[:, 0]), max(positions[:, 0]), n+1)
        self.intervals = np.array([self.boundaries[i:i+2] for i in range(n)])
        self.intervals[:,0] = self.intervals[:,0] - buffer # extend each slice to reduce edge effects
        self.intervals[:,1] = self.intervals[:,1] + buffer
        print(self.intervals)
        self.interpolators = []
        for i in range(n):
            print('Setting up slice {} of {}'.format(i, n))
            subset = get_slice_indices(positions, self.intervals[i,0], self.intervals[i,1])
            interpolator = get_interpolator(positions[subset, :], depths[subset])
            self.interpolators.append(interpolator)

    def __call__(self, *args, **kwargs):
        x = args[0]
        for i in range(len(self.boundaries)-1):
            if self.boundaries[i] <= x <= self.boundaries[i+1]: # have to include both edges
                # print('using ' + str(i))
                return self.interpolators[i](*args, **kwargs)

        print('x = {} outside boundaries {}'.format(x, self.boundaries))


if __name__ == '__main__':
    # print(get_positions(cache, 'MOp1').shape)

    # for id in get_child_area_ids(get_id(structure_tree, 'VIS')):
    #     print(get_acronym(structure_tree, id))


    # get_positions_for_layer('6b') # total 1056
    # get_positions_for_layer('1') # total 10298
    # get_positions_for_layer('6a') # total 12116


    # save_positions_per_layer()

    # save_inner_and_outer_edges()
    edges = load_inner_and_outer_edges()
    # print(edges)
    #
    # inner_edge = edges['inner']
    # outer_edge = edges['outer']
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # plt.plot(inner_edge[:, 0], inner_edge[:, 1], inner_edge[:, 2], 'r.')
    # plt.plot(outer_edge[:, 0], outer_edge[:, 1], outer_edge[:, 2], 'b.')
    # plt.tight_layout()
    # plt.show()

    # save_inner_outer_flags()

    # flags = load_inner_outer_flags()
    #
    positions = load_positions_per_layer()
    positions_all = np.concatenate((
        positions['1'],
        positions['2/3'],
        positions['4'],
        positions['5'],
        positions['6a'],
        positions['6b'],
    ))

    print(positions_all.shape)

    # depths = laplace_solution(positions_all, flags['is_outer'], flags['is_inner'])
    # with open('depths.pkl', 'wb') as file:
    #     pickle.dump(depths, file)

    with open('depths.pkl', 'rb') as file:
        depths = pickle.load(file)
    depths = np.clip(depths, 0, 7)

    colors = np.zeros((len(depths), 3))

    colors[:,0] = depths/7
    colors[:,2] = 1-depths/7

    # plt.hist(depths.flatten(), 100)
    # plt.show()

    # subset = np.random.randint(0, len(depths), 5000)

    # for x in range(min(positions_all[:,0]), max(positions_all[:,0]), 5):
    #     subset = get_slice_indices(positions_all, x, x+5)
    #
    #     fig = plt.figure(figsize=(9, 7))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(positions_all[subset,0], positions_all[subset,1], positions_all[subset,2], c=colors[subset])
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()

    # ci = CompositeInterpolator(positions_all, depths)
    # with open('interpolator.pkl', 'wb') as file:
    #     pickle.dump(ci, file)

    with open('interpolator.pkl', 'rb') as file:
        ci = pickle.load(file)

    # print(ci(24, 35, 76))
    # print(ci(69, 10, 61))
    # print(get_streamline(ci, [69, 10, 61]))


    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(500):
        if i % 50 == 0:
            print('Finding streamline {} of 500'.format(i))
        origin = positions_all[np.random.randint(0, positions_all.shape[0]),:]
        streamline = get_streamline(ci, origin)
        ax.scatter(streamline[0,0], streamline[0,1], streamline[0,2], 'ko')
        ax.plot(streamline[:,0], streamline[:,1], streamline[:,2], 'k-')

        foo = get_closest_voxel(streamline[-1,:], edges['outer'])
        ax.scatter(foo[0], foo[1], foo[2], 'ko')
        # eo = edges['outer']
        # ax.scatter(eo[:, 0], eo[:, 1], eo[:, 2])
        ax.plot(streamline[:,0], streamline[:,1], streamline[:,2], 'k-')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    # min_x = min(positions_all[:,0])
    # max_x = max(positions_all[:,0])
    # print('x range {} to {}'.format(min_x, max_x))
    #
    # subset1 = get_slice_indices(positions_all, 19, 25)
    # print('subset size {} of {}'.format(len(subset1), len(depths)))
    # interpolator1 = get_interpolator(positions_all[subset1, :], depths[subset1])
    # print(interpolator1(24, 35, 76))
    #
    # subset2 = get_slice_indices(positions_all, 70, 103)
    # print('subset size {} of {}'.format(len(subset2), len(depths)))
    # interpolator2 = get_interpolator(positions_all[subset2, :], depths[subset2])
    #
    # test_origin = [24, 35, 76]
    # streamline1 = get_streamline(interpolator1, test_origin, step_size=.2)
    # streamline2 = get_streamline(interpolator2, test_origin, step_size=.1)
    #
    # fig = plt.figure(figsize=(9, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(positions_all[subset2,0], positions_all[subset2,1], positions_all[subset2,2], c=colors[subset2])
    # ax.plot(streamline1[:,0], streamline1[:,1], streamline1[:,2], 'ko-')
    # ax.plot(streamline2[:,0], streamline2[:,1], streamline2[:,2], 'ro-')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # 80, 10, 80 might be a good pivot point for mesh, or 80, 12, 80

    #from testing, need to stay a couple of voxels from edge of volume, .1 voxel steps ok
    #TODO: streamlines from random positions
    #TODO: find outer surface point for each voxel
    #TODO: flatmap



    # with open('interpolator.pkl', 'wb') as file:
    #     pickle.dump(interpolator, file)


    # plt.hist(depths, bins=100)
    # plt.show()


    # print(depths.shape)

    # positions = load_positions_per_layer()
    # # print(positions['1'].shape)
    #
    # # print(get_neighbour_indices(positions['5'][50,:], positions['5']))
    #
    # positions_123 = np.concatenate((positions['1'], positions['2/3']))
    # outer_edge = find_edge(positions['1'], positions_123)
    # print(positions['1'].shape)
    # print(outer_edge.shape)
    #
    # positions_6 = np.concatenate((positions['6a'], positions['6b']))
    # positions_56 = np.concatenate((positions['5'], positions_6))
    # inner_edge = find_edge(positions_6, positions_56)
    # print(positions_56.shape)
    # print(positions_6.shape)
    # print(inner_edge.shape)

    # on_edge(positions['1'][0,:], positions_123)

    # def plot_layer(layer, colour):
    #     p = positions[layer]
    #     plt.plot(p[:,0], p[:,1], p[:,2], colour)
    #
    # fig = plt.figure(figsize=(9, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # # plot_layer('6a', 'k.')
    # plot_layer('6b', 'ro')
    # # plot_layer('5', 'b.')
    # plt.tight_layout()
    # plt.show()

    # print(len(get_acronyms_for_layer('1')))

    # for id in get_leaf_areas():
    #     print('{} ({})'.format(id, get_acronym(structure_tree, id)))

    # for id in get_child_ids(structure_tree, 480149202):
    #     print_descriptions(structure_tree, id)

    # print_descriptions(structure_tree, get_id(structure_tree, 'SSp-bfd'))
    # print_descriptions(structure_tree, get_id(structure_tree, 'AUDd'))
    # print_descriptions(structure_tree, get_id(structure_tree, 'RSPagl'))

