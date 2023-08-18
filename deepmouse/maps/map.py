import numpy as np
from scipy.spatial import ConvexHull
from mcmodels.core import VoxelModelCache
from deepmouse.maps.flatmap import FlatMap
# from maps.flatmap import FlatMap


class Border:
    def __init__(self, area='visual', cache=None):
        if cache is None:
            self.cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
        else:
            self.cache = cache

        self.flatmap = FlatMap.get_instance(area)
        self.flatmap._fit()

    def get_flatmap_border(self, area):
        positions_3d = self._get_positions(area)
        positions_flat = [self.flatmap.get_position_2d(p3d) for p3d in positions_3d]
        hull = ConvexHull(positions_flat)
        return np.array(hull.points[hull.vertices,:])

    def _get_positions(self, area):
        return get_positions(self.cache, area)
        # source_mask = self.cache.get_source_mask()
        # source_keys = source_mask.get_key(structure_ids=None)
        #
        # structure_tree = self.cache.get_structure_tree()
        # id = structure_tree.get_id_acronym_map()[area]
        # mask_indices = np.array(source_mask.mask.nonzero())
        #
        # positions = []
        # for i in range(len(source_keys)):  # single hemisphere
        #     if structure_tree.structure_descends_from(source_keys[i], id):
        #         positions.append(mask_indices[:, i])
        #
        # return np.array(positions)


def get_positions(cache, area, target=False):
    """
    :param cache: VoxelModelCache
    :param area: acronym
    :param target: if True return target voxels, otherwise source voxels (default False)
    :return: positions in order of cache's source mask (note target positions don't match)
    """

    structure_tree = cache.get_structure_tree()
    id = structure_tree.get_id_acronym_map()[area]

    def get_positions_for_mask(mask):
        keys = mask.get_key(structure_ids=None)
        mask_indices = np.array(mask.mask.nonzero())

        positions = []
        for i in range(len(keys)):
            if structure_tree.structure_descends_from(keys[i], id):
                positions.append(mask_indices[:, i])

        return positions

    if target:
        target_positions = get_positions_for_mask(cache.get_target_mask())
        positions = np.array(target_positions)
    else:
        source_positions = get_positions_for_mask(cache.get_source_mask()) # single hemisphere
        positions = np.array(source_positions)

    # target_positions = get_positions_for_mask(cache.get_target_mask())
    # target_positions = np.array(target_positions)
    #
    # foo_positions ends up identical to source_positions
    # min_z = np.min(source_positions[:,2])
    # foo_positions = []
    # for t in target_positions:
    #     if t[2] >= min_z:
    #         foo_positions.append(t)
    # foo_positions = np.array(foo_positions)
    # print(foo_positions.shape)
    #
    # diffs = np.linalg.norm(source_positions - foo_positions, axis=1)
    # print(np.max(diffs))

    return positions


def right_target_indices(cache, area='Isocortex'):
    """
    :return: Indices of target voxels that correspond to source voxels (right hemisphere)
    """
    source_positions = get_positions(cache, area, target=False)
    target_positions = get_positions(cache, area, target=True)

    min_z = np.min(source_positions[:,2])
    test_positions = []
    indices = []
    for i, t in enumerate(target_positions):
        if t[2] >= min_z:
            test_positions.append(t)
            indices.append(i)
    test_positions = np.array(test_positions)

    diffs = np.linalg.norm(source_positions - test_positions, axis=1)

    if np.max(diffs) > 0:
        raise Exception('Source and right-hemisphere target voxel lists do not match')

    return np.array(indices)


class Morph:
    """
    A mapping from one 2D shape to another. Shapes are defined by their
    border contours.
    """

    def __init__(self, source_border, target_border, angle):
        """
        :param source_border: vertices of border of source shape
        :param target_border: vertices of border of target shape
        :param angle: radians to rotate source counter-clockwise to best align with target
        """
        self.source_centre = np.array(find_centre(source_border))
        self.target_centre = np.array(find_centre(target_border))
        self.rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.angles = np.linspace(0, 2*np.pi, 100)

        offsets = np.array(source_border) - self.source_centre
        source_border = np.dot(offsets, self.rot) + self.source_centre

        source_radii = [get_radius(source_border, self.source_centre, angle) for angle in self.angles]
        target_radii = [get_radius(target_border, self.target_centre, angle) for angle in self.angles]
        self.gains = np.divide(target_radii, source_radii)

    def map_to_target(self, source_point):
        offset = np.array(source_point) - self.source_centre
        offset = np.dot(offset, self.rot)
        angle = get_angle(offset)
        gain = np.interp(angle, self.angles, self.gains)
        return self.target_centre + gain * offset


def get_angle(v):
    angle = np.arctan2(v[1], v[0])
    if angle < 0:
        angle = angle + 2*np.pi
    return angle


def get_radius(border, centre, angle):
    result = None
    for i in range(len(border)):
        vertex1 = border[i]
        vertex2 = border[0] if i == len(border) - 1 else border[i+1]
        radius = find_intersection(vertex1, vertex2, centre, angle)
        if radius is not None:
            result = radius
            break
    return result


def find_intersection(vertex1, vertex2, centre, angle):
    # method from https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    result = None

    p = np.array(centre)
    r = np.array([np.cos(angle), np.sin(angle)])
    q = np.array(vertex1)
    s = np.array(vertex2) - q

    rxs = np.cross(r, s)
    u = np.cross(q-p, r) / rxs
    t = np.cross(q - p, s) / rxs
    if not rxs == 0 and 0 <= u <= 1 and t > 0:
        result = t

    return result


def find_centre(border):
    """
    Method from https://en.wikipedia.org/wiki/Centroid#Of_a_polygon

    :param border: ordered points that make up the border
    :return: centre point
    """

    area = 0
    cx = 0
    cy = 0

    for i in range(len(border)):
        x1, y1 = border[i] # this vertex
        x2, y2 = border[i+1] if i < len(border) - 1 else border[0] # next vertex

        d = x1*y2 - x2*y1
        area += d
        cx += (x1+x2) * d
        cy += (y1+y2) * d

    area = area / 2
    cx = cx / (6*area)
    cy = cy / (6*area)

    return np.array((cx, cy))

def is_inside(border, point):
    """
    Method from http://web.archive.org/web/20110314030147/http://paulbourke.net/geometry/insidepoly/

    :param border: ordered points that make up the border
    :param point: a point
    :return: True if point is inside polygon defined by border
    """
    x, y = point
    inside = False
    for i in range(len(border)):
        x1, y1 = border[i]
        x2, y2 = border[i + 1] if i < len(border) - 1 else border[0]

        if ((y1 <= y < y2) or (y2 <= y < y1)) \
                and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside

    return inside


class Field:
    def __init__(self, border, contours, range):
        self.border = border
        self.levels = np.linspace(range[0], range[1], len(contours))

        self.curves = []
        for contour in contours:
            self.curves.append(Curve(contour))

    def get_value(self, point):
        if is_inside(self.border, point):
            distances = [curve.distance(point) for curve in self.curves]

            closest = np.argmin(distances)
            second_closest = _find_second_closest(distances, closest)

            shortest_distance = distances[closest]
            total_distance = distances[closest] + distances[second_closest]
            result = self.levels[closest] + shortest_distance/total_distance \
                     * (self.levels[second_closest]-self.levels[closest])
        else:
            result = np.nan

        return result


def _find_second_closest(distances, closest):
    # find second-closest point among immediate neighbours of closest point

    if closest == 0:
        result = 1
    elif closest == len(distances) - 1:
        result = len(distances) - 2
    elif distances[closest - 1] < distances[closest + 1]:
        result = closest - 1
    else:
        result = closest + 1

    return result


class Curve():
    def __init__(self, points):
        self.points = np.array(points)

    def distance(self, point):
        point_distances = np.linalg.norm(point - self.points, axis=1)
        closest = np.argmin(point_distances)
        second_closest = _find_second_closest(point_distances, closest)

        a = point - self.points[closest]
        b = self.points[second_closest] - self.points[closest]
        return np.abs(np.cross(a, b) / np.linalg.norm(b))


if __name__ == '__main__':
    cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
    sp = get_positions(cache, 'Isocortex')
    print(sp.shape)
    tp = get_positions(cache, 'Isocortex', target=True)
    print(tp.shape)

    r = right_target_indices(cache)

    foo = tp[r,:]
    print(foo.shape)

