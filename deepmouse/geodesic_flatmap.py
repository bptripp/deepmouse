import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gdist
from shapely.geometry import Polygon, Point
# from deepmouse.maps.map import right_target_indices
# from deepmouse.maps.util import ResultCache
# from deepmouse.maps.util import get_voxel_model_cache, get_positions, get_id, get_default_structure_tree
from maps.map import right_target_indices
from maps.util import ResultCache
from maps.util import get_voxel_model_cache, get_positions, get_id, get_default_structure_tree


cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()


def cross_prod(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])


def left_turn(p0, p1, p2):
    return cross_prod(p0, p1, p2) > 0


def convex_hull(points):
    # https://sandipandey.wixsite.com/simplydatascience/post/implementing-a-few-advanced-algorithms-with-python
    n = len(points)
    # the first point
    _, p0 = min([((x, y), i) for i, (x, y) in enumerate(points)])
    start_index = p0
    print(p0)

    def close_enough(p0, p1):
        distance = (
            (points[p0][0] - points[p1][0]) ** 2 + (points[p0][1] - points[p1][1]) ** 2
        ) ** 0.5
        return p0 != p1 and distance < 10

    chull = []
    # while (True):
    for i in range(50):
        chull.append(p0)
        p1 = (p0 + 1) % n  # make sure p1 != p0

        for p2 in range(n):
            if left_turn(points[p0], points[p1], points[p2]):
                p1 = p2

        p0 = p1
        # came back to first point?
        if p0 == start_index:
            break

    return chull


def concave_hull(points):
    chull = convex_hull(points)

    def dist(pa, pb):
        return (
            (points[pa][0] - points[pb][0]) ** 2 + (points[pa][1] - points[pb][1]) ** 2
        ) ** 0.5

    def between(pa, pb, pc):
        # True if projection of pc onto pa-pb is between pa and pb
        rel_b = np.array(points[pb]) - np.array(points[pa])
        unit_b = rel_b / np.linalg.norm(rel_b)
        rel_c = np.array(points[pc]) - np.array(points[pa])
        scalar_projection = np.dot(rel_c, unit_b)
        return scalar_projection > 0 and scalar_projection < dist(pa, pb)

    def concavity(pa, pb, pc):
        # concavity of curve through pc relative to line from pa to pb
        # note concavity is rate of change of derivative
        ab_distance = dist(pa, pb)
        ab_angle = np.arctan2(
            points[pb][1] - points[pa][1], points[pb][0] - points[pa][0]
        )
        ac_distance = dist(pa, pc)
        ac_angle = np.arctan2(
            points[pc][1] - points[pa][1], points[pc][0] - points[pa][0]
        )

        # rotate and scale as if pa and pb are at -1,0 and 1,0
        normalized_angle = ac_angle - ab_angle
        normalized_distance = ac_distance / ab_distance
        x = -1 + 2 * normalized_distance * np.cos(normalized_angle)
        y = 2 * normalized_distance * np.sin(normalized_angle)

        # solve y = alpha x^2 - alpha for alpha
        return y / (x ** 2 - 1)  # scale of the parabola on which pc lies

    def next(index):
        if index == len(chull) - 1:
            return 0
        else:
            return index + 1

    distances = []
    for i in range(len(chull)):
        distances.append(dist(i, next(i)))

    mean_distance = np.mean(distances)

    def find_least_concave_point(pa, pb, max_concavity=1):
        not_in_hull = np.setdiff1d(range(len(points)), chull)

        least_concave_point = None
        least_convavity = None
        for pc in not_in_hull:
            if between(pa, pb, pc):
                c = concavity(pa, pb, pc)
                if c > 0 and c <= max_concavity:
                    if least_concave_point == None or c < least_convavity:
                        least_concave_point = pc
                        least_convavity = c

        return least_concave_point

    def add_concavities():
        print("len hull {}".format(len(chull)))
        added_something = False
        i = 0
        while i < len(chull):
            if (
                dist(chull[i], chull[next(i)]) > 2 * mean_distance
            ):  # consider concave intermediate points
                pc = find_least_concave_point(chull[i], chull[next(i)])
                if pc:
                    added_something = True
                    chull.insert(i + 1, pc)
                    i = i + 1
                    print("*", end="", flush=True)
                else:
                    print(".", end="", flush=True)
            i = i + 1
        print("")
        return added_something

    while add_concavities():
        pass

    return chull


# def find_middle(area):
#     """
#     :param area: name of cortical area
#     :return: average of 3D voxel coordinates
#     """
#     id = get_id(structure_tree, area)
#     positions = get_positions(cache, id)
#     middle = np.mean(positions)


class GeodesicFlatmap:
    def __init__(self, area=None):
        data = ResultCache.get("mesh-data")
        vertices = data["vertices"].astype(float)
        triangles = data["triangles"].astype("int32")

        vs = ResultCache.get("voxel-to-surface")
        self.voxel_positions = vs["voxel_positions"]  # Isocortex, source order
        self.surface_indices = vs["surface_indices"]

        if area is None:
            reference_coords = [65, 5, 75]
        else:
            id = get_id(structure_tree, area)
            positions = get_positions(cache, id)
            middle = np.mean(positions, axis=0)
            vertex_distances = np.sum((vertices - middle) ** 2, axis=1)
            closest_index = np.argmin(vertex_distances)
            reference_coords = vertices[closest_index]
            # distances = np.sum((self.voxel_positions - middle)**2, axis=1)
            # closest_index = np.argmin(distances)
            # surface_index = self.surface_indices[closest_index]
            # reference_coords = self.voxel_positions[surface_index]

        reference = np.where((vertices == reference_coords).all(axis=1))[0].astype(
            "int32"
        )
        reference_dist = gdist.compute_gdist(
            vertices, triangles, source_indices=reference
        )

        angles = np.arctan2(
            vertices[:, 2] - vertices[reference, 2],
            vertices[:, 0] - vertices[reference, 0],
        )
        self.ap_position = -np.cos(angles) * reference_dist
        self.ml_position = np.sin(angles) * reference_dist
        self.vertices = vertices

        # print(reference_dist.shape)
        # print(self.voxel_positions.shape)
        # print(len(self.surface_indices))
        # print(self.surface_indices)

        self.voxel_colours = None
        self.clear_voxel_colours()
        self.voxel_vectors = None

    def clear_voxel_colours(self):
        self.voxel_colours = np.zeros((len(self.surface_indices), 3))

    def set_voxel_colour(self, voxel_position, rgb):
        ind = np.where((self.voxel_positions == voxel_position).all(axis=1))[0]
        if len(ind) == 0:
            raise Exception("Unknown voxel: {}".format(voxel_position))
        self.voxel_colours[ind, :] = rgb

    def get_voxel_colour(self, voxel_position):
        ind = np.where((self.voxel_positions == voxel_position).all(axis=1))[0]
        if len(ind) == 0:
            raise Exception("Unknown voxel: {}".format(voxel_position))
        return self.voxel_colours[ind, :]

    def clear_voxel_vectors(self, dim):
        self.voxel_vectors = np.zeros((len(self.surface_indices), dim))

    def set_voxel_vector(self, voxel_position, vector):
        ind = np.where((self.voxel_positions == voxel_position).all(axis=1))[0]
        if len(ind) == 0:
            raise Exception("Unknown voxel: {}".format(voxel_position))
        self.voxel_vectors[ind] = vector

    def get_voxel_vector(self, voxel_position):
        ind = np.where((self.voxel_positions == voxel_position).all(axis=1))[
            0
        ]  # TODO: extract get_ind
        if len(ind) == 0:
            raise Exception("Unknown voxel: {}".format(voxel_position))
        return self.voxel_vectors[ind[0], :]

    def get_voxel_index(self, voxel_position):
        ind = np.where((self.voxel_positions == voxel_position).all(axis=1))[
            0
        ]  # TODO: extract get_ind
        if len(ind) == 0:
            raise Exception("Unknown voxel: {}".format(voxel_position))
        return ind[0]

    def get_position_2d(self, voxel_position):
        voxel_index = self.get_voxel_index(voxel_position)
        surface_index = self.surface_indices[voxel_index]
        return [self.ml_position[surface_index], self.ap_position[surface_index]]

    def show_map(self, image_file=None):
        flatmap_sums = np.zeros((self.ml_position.shape[0], 3))
        flatmap_counts = np.zeros(self.ml_position.shape[0]) + 1e-6

        for (voxel_colour, surface_index) in zip(
            self.voxel_colours, self.surface_indices
        ):
            flatmap_sums[surface_index, :] = (
                flatmap_sums[surface_index, :] + voxel_colour
            )
            flatmap_counts[surface_index] = flatmap_counts[surface_index] + 1

        flatmap_colours = (flatmap_sums.T / flatmap_counts).T

        # create pixel image
        width_pixels = 200
        border_pixels = 3
        ml_range = max(self.ml_position) - min(self.ml_position)
        ap_range = max(self.ap_position) - min(self.ap_position)
        height_pixels = int(width_pixels * ap_range / ml_range)

        from scipy import interpolate

        x = np.zeros((len(self.ml_position), 2))
        x[:, 0] = self.ml_position
        x[:, 1] = self.ap_position

        pos_x = np.linspace(min(self.ml_position), max(self.ml_position), width_pixels)
        pos_y = np.linspace(min(self.ap_position), max(self.ap_position), height_pixels)
        grid_x = np.meshgrid(pos_x, pos_y)
        flat_x = np.array((grid_x[0].flatten(), grid_x[1].flatten())).T

        image = np.zeros((height_pixels, width_pixels, 3))
        for i in range(3):
            channel_interpolator = interpolate.RBFInterpolator(
                x, flatmap_colours[:, i], neighbors=4, kernel="gaussian", epsilon=1
            )
            channel_flat = channel_interpolator(flat_x)
            channel_grid = channel_flat.reshape(height_pixels, width_pixels)
            channel_grid = np.flipud(channel_grid)
            image[:, :, i] = channel_grid
            print(channel_grid.shape)

        chull = ResultCache.get("chull", function=lambda: concave_hull(x))
        s = Polygon(x[chull, :])

        mask = np.zeros(flat_x.shape[0])
        for i, fx in enumerate(flat_x):
            if s.contains(Point(fx[0], fx[1])):
                mask[i] = 1
        mask = mask.reshape(height_pixels, width_pixels)
        mask = np.flipud(mask)
        for i in range(3):
            image[:, :, i] = 1 - (1 - image[:, :, i]) * mask

        # interpolate.RBFInterpolator()
        # red = interpolate.interp2d(self.ml_position, self.ap_position, flatmap_colours[:,0], kind='cubic')
        # green = interpolate.interp2d(self.ml_position, self.ap_position, flatmap_colours[:,1], kind='cubic')
        # blue = interpolate.interp2d(self.ml_position, self.ap_position, flatmap_colours[:,2], kind='cubic')
        # new_ml = np.linspace(min(self.ml_position), max(self.ml_position), width_pixels)
        # new_ap = np.linspace(min(self.ap_position), max(self.ap_position), height_pixels)
        # red_new = red(new_ml, new_ap)
        # green_new = green(new_ml, new_ap)
        # blue_new = blue(new_ml, new_ap)
        # print(red_new.shape)
        #
        # image = np.zeros((height_pixels, width_pixels, 3))
        # image[:,:,0] = red_new
        # image[:,:,1] = green_new
        # image[:,:,2] = blue_new

        # plt.subplot(122)
        plt.figure(figsize=(3, 3))
        plt.imshow(image)
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        if image_file:
            plt.savefig(image_file)

    def draw_boundary(self, area):
        id = structure_tree.get_id_acronym_map()[area]
        area_positions = get_positions(cache, id)
        area_positions_2d = np.array([self.get_position_2d(p) for p in area_positions])
        hull = concave_hull(area_positions_2d)
        hull.append(hull[0])  # close the boundary
        hull.append(hull[1])  # extra points to avoid edge effects when filtering
        hull.append(hull[2])
        boundary = area_positions_2d[hull, :]
        boundary[:, 0] = np.convolve(boundary[:, 0], [1 / 3, 1 / 3, 1 / 3], "same")
        boundary[:, 1] = np.convolve(boundary[:, 1], [1 / 3, 1 / 3, 1 / 3], "same")
        boundary = boundary[1:-1, :]  # remove extra points

        width_pixels = 200  # TODO: redundant with code in show_map
        ml_range = max(self.ml_position) - min(self.ml_position)
        ap_range = max(self.ap_position) - min(self.ap_position)
        height_pixels = int(width_pixels * ap_range / ml_range)
        image_boundary = np.zeros_like(boundary)
        image_boundary[:, 0] = (
            (boundary[:, 0] - min(self.ml_position)) / ml_range * width_pixels
        )
        image_boundary[:, 1] = (
            (boundary[:, 1] - min(self.ap_position)) / ap_range * height_pixels
        )
        image_boundary[:, 1] = height_pixels - image_boundary[:, 1]

        plt.plot(image_boundary[:, 0], image_boundary[:, 1], "w")


# test poly_area ...
# angles = np.arange(0, 2*np.pi, .001)
# print(angles)
# r = (1/np.pi)**.5
# x = r * np.cos(angles)
# y = r * np.sin(angles)

# x = np.array([0, 10, 10, 0])
# y = np.array([0, 0, 10, 10])
#
# print(poly_area(x, y))
#
# plt.plot(x, y)
# plt.show()

# shrink_wrap(transformed[:,0], transformed[:,1], 1)
# hull = convex_hull(transformed)
# print(len(hull))
# print(transformed.shape)
# print(transformed[hull,:])
# plt.plot(transformed[hull,0], transformed[hull,1])
# plt.scatter(transformed[:,0], transformed[:,1])
# plt.show()

# test concave hull
# foo = concave_hull(transformed)
# plt.plot(transformed[foo,0], transformed[foo,1], 'r')
# plt.scatter(transformed[:,0], transformed[:,1])
# plt.show()


# def get_weights(source_name, target_name, data_folder='data_files/'):
#     source_mask = cache.get_source_mask()
#     source_keys = source_mask.get_key(structure_ids=None)
#     weight_file = data_folder + '/voxel-weights.pkl'
#     node_file = data_folder + '/voxel-nodes.pkl'
#     if os.path.isfile(weight_file) and os.path.isfile(node_file):
#         with open(weight_file, 'rb') as file:
#             weights = pickle.load(file)
#         with open(node_file, 'rb') as file:
#             nodes = pickle.load(file)
#
#     pre_id = structure_tree.get_id_acronym_map()[source_name]
#     post_id = structure_tree.get_id_acronym_map()[target_name]
#
#     pre_indices = []
#     post_indices = []
#
#     for i in range(len(source_keys)):
#         if structure_tree.structure_descends_from(source_keys[i], pre_id):
#             pre_indices.append(i)
#         if structure_tree.structure_descends_from(source_keys[i], post_id):
#             post_indices.append(i)
#
#     weights_by_target_voxel = []
#     for pi in post_indices:
#         w = np.dot(weights[pre_indices, :], nodes[:, pi])
#         weights_by_target_voxel.append(w)
#     return weights_by_target_voxel


def propagate_through_isocortex(
    vectors, data_folder="data_files/", ignore_zero_vectors=False
):
    """
    :param vectors: An initial vector for each isocortex voxel
    :return: Vectors multiplied by weights of connections between isocortex voxels
    """

    # weight_file = data_folder + '/voxel-weights.pkl'
    # node_file = data_folder + '/voxel-nodes.pkl'

    # if os.path.isfile(weight_file) and os.path.isfile(node_file):
    #     with open(weight_file, 'rb') as file:
    #         weights = pickle.load(file)
    #     with open(node_file, 'rb') as file:
    #         nodes = pickle.load(file)
    # else:
    #     raise Exception('Weight files missing')
    weights = cache.get_weights()
    nodes = cache.get_nodes()
    # print(weights.shape) # source to latent (226346, 428)
    # print(nodes.shape) # latent to target (both hemispheres) (428, 448962)

    cortex_id = structure_tree.get_id_acronym_map()["Isocortex"]

    source_mask = cache.get_source_mask()
    source_keys = source_mask.get_key(
        structure_ids=None
    )  # structure ids for all voxels
    print("len source keys {}".format(len(source_keys)))
    # 
    source_cortex_indices = []
    for i in range(len(source_keys)):
        if structure_tree.structure_descends_from(source_keys[i], cortex_id):
            source_cortex_indices.append(i)

    target_mask = cache.get_target_mask()
    target_keys = target_mask.get_key(
        structure_ids=None
    )  # structure ids for all voxels
    print("len target keys {}".format(len(target_keys)))

    target_cortex_indices = []
    for i in range(len(target_keys)):
        if structure_tree.structure_descends_from(target_keys[i], cortex_id):
            target_cortex_indices.append(i)

    target_cortex_indices = np.array(target_cortex_indices)
    r = right_target_indices(cache)
    right_target_cortex_indices = target_cortex_indices[r]

    cortex_weights = weights[source_cortex_indices, :]
    cortex_nodes = nodes[:, right_target_cortex_indices]

    # calculate sum of weights inbound to each target voxel for normalization
    weights_to_include = np.ones(cortex_weights.shape[0])
    if ignore_zero_vectors:
        zero_vectors = np.where((vectors == 0).all(axis=1))[0]
        weights_to_include[zero_vectors] = 0
    sums = np.dot(cortex_nodes.T, np.dot(cortex_weights.T, weights_to_include))

    # note: wide range of these so we should normalize voxel-wise, otherwise won't see much
    # plt.hist(sums)
    # plt.show()

    result = np.dot(cortex_nodes.T, np.dot(cortex_weights.T, vectors))
    mean_sum = np.mean(sums)
    
    for i in range(len(result)):
        if sums[i] > 0:
            if ignore_zero_vectors:
                result[i] = result[i] / (sums[i] + mean_sum / 10)
            else:
                result[i] = result[i] / sums[i]  # weighted average

    # result = []
    # for i, ci in enumerate(cortex_indices):
    #     if i % 100 == 0:
    #         print('{} of {}'.format(i, len(cortex_indices)))
    #
    #     w = np.dot(weights[cortex_indices, :], nodes[:, ci])
    #     # w = np.dot(weights[foo_indices, :], nodes[:, ci]) #TODO: check this is right direction
    #     ws = np.sum(w)
    #     if ws > 0:
    #         w = w / ws # normalize so that sum = 1
    #     voxel_result = np.dot(w, vectors)
    #     print(voxel_result)
    #
    #     result.append(voxel_result)

    print("{} result".format(len(result)))

    return np.array(result)


def set_nan_to_zero(number):
    if not number:
        return 0
    elif np.isnan(number):
        return 0
    else:
        return number


def get_probe_area_colours():
    means = []
    sds = []
    for area in probe_areas:
        id = structure_tree.get_id_acronym_map()[area]
        ps = get_positions(cache, id)
        vectors = []
        for p in ps:
            vectors.append(flatmap.get_voxel_colour(p))
        vectors = np.array(vectors)
        # 
        means.append(np.mean(vectors, axis=0)[0])
        sds.append(np.std(vectors, axis=0)[0])

    return np.array(means), np.array(sds)


def get_probe_area_vector_corr():
    print("correlations:")
    corrs = []
    for area in probe_areas:
        id = structure_tree.get_id_acronym_map()[area]
        ps = get_positions(cache, id)
        vectors = []
        for p in ps:
            vectors.append(flatmap.get_voxel_vector(p))
        vectors = np.array(vectors)

        # TODO: ignore rows with zeros?
        vectors_with_zeros = np.where((vectors == 0).any(axis=1))[0]
        print(
            "{} of {} vectors with zeros".format(
                len(vectors_with_zeros), vectors.shape[0]
            )
        )

        corr = np.corrcoef(vectors, rowvar=False)
        print(corr)
        corrs.append(corr)

    return corrs


if __name__ == "__main__":
    probe_areas = [
        "VISal",
        "VISam",
        "VISl",
        "VISp",
        "VISpl",
        "VISpm",
        "VISli",
        "VISpor",
        "VISa",
        "VISrl",
    ]

    # plot results of probe
    # all_means = np.zeros((6, 10, 3))
    # all_sds = np.zeros((6, 10, 3))
    # for i in range(6):
    #     with open('step_{}_probe.pkl'.format(i), 'rb') as file:
    #         probe_areas, means, sds = pickle.load(file)

    #     all_means[i,:,:] = means
    #     all_sds[i,:,:] = sds

    # plt.figure(figsize=(8,4))
    # for i in range(10):
    #     plt.subplot(2,5,i+1)
    #     plt.title(probe_areas[i])
    #     means = all_means[:,i,:]
    #     sds = all_sds[:,i,:]
    #     plt.errorbar(range(6), means[:,0], yerr=sds[:,0], capsize=3, fmt='r.')
    #     plt.errorbar(range(6), means[:,1], yerr=sds[:,1], capsize=3, fmt='g.')
    #     plt.errorbar(range(6), means[:,2], yerr=sds[:,2], capsize=3, fmt='b.')
    #     plt.ylim([0, 1])
    # plt.subplot(2,5,1), plt.ylabel('Fraction Input')
    # plt.subplot(2,5,6), plt.ylabel('Fraction Input')
    # plt.subplot(2,5,8), plt.xlabel('Steps')
    # plt.tight_layout()
    # plt.savefig('mix-per-step.png')
    # plt.show()

    # exit()
    flatmap = GeodesicFlatmap()

    # id = get_id(structure_tree, 'Isocortex')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     flatmap.set_voxel_colour(position, [0, 0, .2])

    from deepmouse.maps.visual import VoxelRetinotopy
    from deepmouse.maps.auditory import FrequencyMap
    from deepmouse.maps.barrel import WhiskerMap

    ## propagate topography ...
    # voxel_retinotopy = VoxelRetinotopy(cache)
    # id = get_id(structure_tree, 'VISp')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     azimuth, altitude = voxel_retinotopy.get_retinal_coords(position)
    #     flatmap.set_voxel_colour(position,
    #                              [(set_nan_to_zero(azimuth)+60)/120, 1, (set_nan_to_zero(altitude)+40)/80])

    # whisker_map = WhiskerMap(cache)
    # whisker_map.parameterize()
    # id = get_id(structure_tree, 'SSp-bfd')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     azimuth = whisker_map.get_azimuth(position)
    #     elevation = whisker_map.get_elevation(position)
    #     flatmap.set_voxel_colour(position, [set_nan_to_zero(azimuth)/9, 1, set_nan_to_zero(elevation)/5])

    # frequency_map = FrequencyMap(cache)
    # id = get_id(structure_tree, 'AUDp')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     log_pf = np.log(frequency_map.get_preferred_frequency(position))
    #     fraction = (log_pf - np.log(3000)) / (np.log(30000) - np.log(3000))
    #     flatmap.set_voxel_colour(position, [set_nan_to_zero(fraction), 1, 0])

    # # propagate vectors with full topography ...
    # flatmap.clear_voxel_vectors(5)
    # voxel_retinotopy = VoxelRetinotopy(cache)
    # id = get_id(structure_tree, 'VISp')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     azimuth, altitude = voxel_retinotopy.get_retinal_coords(position)
    #     vv = flatmap.get_voxel_vector(position)
    #     vv[0] = set_nan_to_zero(azimuth)
    #     vv[1] = set_nan_to_zero(altitude)
    #     flatmap.set_voxel_vector(position, vv)
    #
    # whisker_map = WhiskerMap(cache)
    # whisker_map.parameterize()
    # id = get_id(structure_tree, 'SSp-bfd')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     azimuth = whisker_map.get_azimuth(position)
    #     elevation = whisker_map.get_elevation(position)
    #     vv = flatmap.get_voxel_vector(position)
    #     vv[2] = set_nan_to_zero(azimuth)
    #     vv[3] = set_nan_to_zero(elevation)
    #     flatmap.set_voxel_vector(position, vv)
    #
    # frequency_map = FrequencyMap(cache)
    # id = get_id(structure_tree, 'AUDp')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     log_pf = np.log(frequency_map.get_preferred_frequency(position))
    #     vv = flatmap.get_voxel_vector(position)
    #     vv[4] = set_nan_to_zero(log_pf)
    #     flatmap.set_voxel_vector(position, vv)
    #
    # # subtract mean to avoid correlation via changes in strength
    # for i in range(5):
    #     print('subtracting {}'.format(np.mean(flatmap.voxel_vectors[:,i])))
    #     flatmap.voxel_vectors[:,i] = flatmap.voxel_vectors[:,i] - np.mean(flatmap.voxel_vectors[:,i])

    # set areas to a solid colour ...
    id = get_id(structure_tree, "VISp")
    # id = get_id(structure_tree, 'RSPagl')
    positions = get_positions(cache, id)
    for position in positions:
        flatmap.set_voxel_colour(position, [1, 0, 0])
    # id = get_id(structure_tree, 'RSPd')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     flatmap.set_voxel_colour(position, [1, 0, 0])
    # id = get_id(structure_tree, 'RSPv')
    # positions = get_positions(cache, id)
    # for position in positions:
    #     flatmap.set_voxel_colour(position, [1, 0, 0])

    # id = get_id(structure_tree, 'SSp-bfd')
    id = get_id(structure_tree, "SSp")
    # id = get_id(structure_tree, 'ACA')
    positions = get_positions(cache, id)
    for position in positions:
        flatmap.set_voxel_colour(position, [0, 1, 0])

    id = get_id(structure_tree, "AUDp")
    # id = get_id(structure_tree, 'MOp')
    # id = get_id(structure_tree, 'TEa')
    positions = get_positions(cache, id)
    for position in positions:
        flatmap.set_voxel_colour(position, [0, 0, 1])

    step_i = flatmap.voxel_colours
    # step_i = flatmap.voxel_vectors # **************
    with open("step_0.pkl", "wb") as file:
        pickle.dump(step_i, file)

    means, sds = get_probe_area_colours()
    with open("step_0_probe.pkl", "wb") as file:
        pickle.dump((probe_areas, means, sds), file)
    # corrs = get_probe_area_vector_corr() # *******************
    # with open('step_0_probe_corr.pkl', 'wb') as file:
    #     pickle.dump((probe_areas, corrs), file)

    # flatmap.show_map(image_file='step_0.png')
    # plt.show()

    for i in range(1, 2):
        step_i = propagate_through_isocortex(step_i)
        # step_i = propagate_through_isocortex(step_i, ignore_zero_vectors=True) #*********
        # print(step_i)
        # flatmap.voxel_vectors = step_i # *************************
        flatmap.voxel_colours = 2 * step_i

        with open("step_{}.pkl".format(i), "wb") as file:
            pickle.dump(step_i, file)

        means, sds = get_probe_area_colours()
        with open("step_{}_probe.pkl".format(i), "wb") as file:
            pickle.dump((probe_areas, means, sds), file)
        # corrs = get_probe_area_vector_corr() # *******************
        # with open('step_{}_probe_corr.pkl'.format(i), 'wb') as file:
        #     pickle.dump((probe_areas, corrs), file)

        flatmap.show_map(image_file="step_{}.png".format(i))

    # TODO: draw lines around VISp and other visual areas?
