"""
Barrel cortex map from:

﻿C. C. H. Petersen, “The Functional Organization of the Barrel Cortex,” Neuron, vol. 56, pp. 339–355, 2007.

The shape of barrel cortex in the Allen atlas differs from the outline of the barrels in various publications,
e.g. Petersen (above), also McCasland & Woolsey (1988) J Comp Neurol. We will assume the barrels fall within
the barrel cortex, and that the barrel cortex has some additional corners that don't contain barrels.
"""

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from mcmodels.core import VoxelModelCache
from deepmouse.maps.map import Border, is_inside, get_positions


#TODO: clean up
#TODO: consider thinning motor cortex
#TODO: barrels show up clerly in average volume but not in Nissl volume that we can download; maybe take from Harris et al. figure
#TODO: use streamlines rather than nearest L4 voxel

class WhiskerMap:
    """
    Estimates map of whisker identity for barrel-cortex voxels.
    """
    def __init__(self, cache, plot_borders=False):
        # self.border = Border(area='barrel') #TODO: clean up
        # self._l4_positions = self.border._get_positions('SSp-bfd4')
        self.positions = get_positions(cache, 'SSp-bfd')
        self._l4_positions = get_positions(cache, 'SSp-bfd4')
        self.centers = None
        self.border = None

        stacks = {}

        def make_key(position):
            return '{},{}'.format(position[0],position[2])

        for p in self._l4_positions:
            key = make_key(p)
            if not key in stacks:
                stacks[key] = [p]
            else:
                stacks[key].append(p)

        thin_positions = []
        for key in stacks.keys():
            p = np.array(stacks[key])
            thin_positions.append(np.mean(p, axis=0))
        self._l4_positions = np.array(thin_positions)

    def set_barrel_centers(self, centers):
        self.centers = centers

    def set_border(self, border):
        self.border = border

    def set_azimuth_elevation(self, azimuth, elevation):
        # Estimates for L4
        if self.centers is None or self.border is None:
            raise Exception('Call set_barrel_centers() and set_border() first')

        self.azimuth_function = interp2d(self.centers[:,0], self.centers[:,1], azimuth, kind='cubic')
        self.elevation_function = interp2d(self.centers[:,0], self.centers[:,1], elevation, kind='cubic')

    def get_azimuth(self, position):
        closest = self._find_closest_l4_position(position)
        if self._inside_border(closest):
            return self.azimuth_function(closest[0], closest[2])[0]
        else:
            return None

    def get_elevation(self, position):
        closest = self._find_closest_l4_position(position)
        if self._inside_border(closest):
            return self.elevation_function(closest[0], closest[2])[0]
        else:
            return None

    def _find_closest_l4_position(self, position):
        l4_distances = np.linalg.norm(position - self._l4_positions, axis=1)
        return self._l4_positions[np.argmin(l4_distances), :]

    def _inside_border(self, position):
        point = (position[0], position[2])
        return is_inside(self.border, point)

    def parameterize(self):
        border = np.array(barrel_border)
        centres = np.array(barrel_centres)
        border[:, 0] = -border[:, 0]
        centres[:, 0] = -centres[:, 0]
        angle = -.55
        scale_factor = 18.25
        offset = (70.75, 77.125)
        border = transform(border, angle, scale_factor, offset)
        centres = transform(centres, angle, scale_factor, offset)
        self.set_barrel_centers(centres)
        self.set_border(border)
        ae = np.array(barrel_azimith_elevation)
        self.set_azimuth_elevation(ae[:, 0], ae[:, 1])


# Barrel centres in this order:
# alpha, beta, gamma, delta, a1-4, b1-4, c1-8, d1-8, e1-9
barrel_centres = [
    [0.216551724137931, 0.8034321372854913],
    [0.1668965517241379, 0.6833073322932917],
    [0.14344827586206896, 0.5522620904836193],
    [0.1875862068965517, 0.38533541341653654],
    [0.3379310344827586, 0.8580343213728547],
    [0.47310344827586204, 0.8471138845553821],
    [0.5889655172413792, 0.8283931357254288],
    [0.6855172413793104, 0.7925117004680187],
    [0.31448275862068964, 0.7301092043681746],
    [0.4372413793103448, 0.7176287051482058],
    [0.5517241379310345, 0.7098283931357253],
    [0.6482758620689655, 0.6989079563182525],
    [0.2924137931034482, 0.5429017160686427],
    [0.4234482758620689, 0.5304212168486738],
    [0.5282758620689655, 0.549141965678627],
    [0.6151724137931034, 0.5413416536661466],
    [0.6882758620689655, 0.5756630265210607],
    [0.7544827586206896, 0.6068642745709827],
    [0.8110344827586206, 0.6365054602184086],
    [0.8744827586206896, 0.6677067082683307],
    [0.3379310344827586, 0.3572542901716068],
    [0.46620689655172415, 0.3525741029641184],
    [0.5682758620689654, 0.3525741029641184],
    [0.656551724137931, 0.3385335413416536],
    [0.7144827586206897, 0.3978159126365054],
    [0.7655172413793102, 0.4524180967238688],
    [0.8110344827586206, 0.5085803432137285],
    [0.8717241379310343, 0.5538221528861154],
    [0.3337931034482759, 0.19188767550702024],
    [0.47172413793103446, 0.16692667706708264],
    [0.5834482758620688, 0.1497659906396256],
    [0.6689655172413793, 0.14040561622464898],
    [0.7393103448275861, 0.16068642745709827],
    [0.7958620689655171, 0.20280811232449292],
    [0.8482758620689654, 0.2527301092043681],
    [0.8841379310344827, 0.32605304212168473],
    [0.9310344827586207, 0.40093603744149753]
]

# a1-4, b1-4, c1-8, d1-8, e1-9
barrel_azimith_elevation = [
    [0, 3.5], # alpha
    [0, 2.5], # beta
    [0, 1.5], # gamma
    [0, 0.5], # delta
    [1, 5], # a1-4
    [2, 5],
    [3, 5],
    [4, 5],
    [1, 4], # b1-4
    [2, 4],
    [3, 4],
    [4, 4],
    [1, 3], #c1-8
    [2, 3],
    [3, 3],
    [4, 3],
    [5, 3],
    [6, 3],
    [7, 3],
    [8, 3],
    [1, 2], # d1-8
    [2, 2],
    [3, 2],
    [4, 2],
    [5, 2],
    [6, 2],
    [7, 2],
    [8, 2],
    [1, 1], # e1-9
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 1],
    [6, 1],
    [7, 1],
    [8, 1],
    [9, 1]
]

barrel_border = [
    [0.8965517241379309, 0.7098283931357253],
    [0.8013793103448276, 0.6817472698907955],
    [0.7393103448275861, 0.6661466458658345],
    [0.6868965517241379, 0.641185647425897],
    [0.6813793103448276, 0.6801872074882995],
    [0.7268965517241379, 0.7737909516380654],
    [0.713103448275862, 0.8237129485179406],
    [0.6827586206896551, 0.8424336973478938],
    [0.6193103448275862, 0.8658346333853353],
    [0.4868965517241379, 0.8939157566302651],
    [0.37103448275862067, 0.9063962558502339],
    [0.313103448275862, 0.9079563182527299],
    [0.20137931034482756, 0.862714508580343],
    [0.16965517241379308, 0.7862714508580342],
    [0.10344827586206896, 0.6817472698907955],
    [0.08551724137931034, 0.5351014040561621],
    [0.1268965517241379, 0.32137285491419654],
    [0.30344827586206896, 0.12636505460218395],
    [0.5820689655172414, 0.06552262090483607],
    [0.7241379310344827, 0.06708268330733236],
    [0.7751724137931034, 0.07332293291731662],
    [0.8441379310344828, 0.11544461778471138],
    [0.8910344827586206, 0.17004680187207488],
    [1.0013793103448274, 0.329173166926677],
    [0.9972413793103447, 0.38065522620904824],
    [0.9517241379310344, 0.4243369734789392],
    [0.9213793103448275, 0.46957878315132595],
    [0.9158620689655171, 0.5257410296411855]
]


# def rotate(points, angle):
#     R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#     # result = np.zeros_like(points)
#     # for i in range(points.shape[0]):
#     #     result[i,:] = np.matmul(R, points[i,:])
#     # return result
#     return np.matmul(R, points.T).T
#
#
# def shift(points, offset):
#     return points + offset
#
#
# def scale(points, scale_factor):
#     # mx = np.mean(points[:,0])
#     # my = np.mean(points[:,1])
#     # points = points - (mx, my)
#     points = scale_factor * points
#     return points
#     # return points + (mx, my)


def transform(points, angle, scale, offset):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.matmul(R, points.T).T
    points = scale * points
    return points + offset


def draw_voxel_squares(positions):
    for position in positions:
        x = position[0]
        y = position[2]
        plt.plot([x-.5, x+.5, x+.5, x-.5, x-.5], [y-.5, y-.5, y+.5, y+.5, y-.5], 'k')


def get_barrel(centres, position, map):
    """
    :param centres: barrel centres in ML/AP voxel coordinates
    :param position: 3D voxel position (back, down, lateral)
    :return:
    """
    # l4_positions = map.border._get_positions('SSp-bfd4')
    l4_distances = np.linalg.norm(position - map._l4_positions, axis=1)
    closest_l4_position = map._l4_positions[np.argmin(l4_distances), :]
    closest_l4_position = (closest_l4_position[0], closest_l4_position[2])

    barrel_distances = np.linalg.norm(closest_l4_position - centres, axis=1)
    barrel_index = np.argmin(barrel_distances)

    if barrel_distances[barrel_index] <= 1.5:
        return barrel_index
    else:
        return None


def get_colour(barrel_index):
    if barrel_index is None:
        return [0, 0, 0, 1]
    else:
        cmap = plt.get_cmap("tab20")
        return cmap(barrel_index % cmap.N)


def plot_ae(cache, map):
    positions = map.positions

    source_mask = cache.get_source_mask()
    source_keys = source_mask.get_key(structure_ids=None)
    source_key_volume = source_mask.map_masked_to_annotation(source_keys)
    voxels = source_key_volume < 0 #all False
    voxels[positions[:,0], positions[:,1], positions[:,2]] = True

    colors = np.empty(voxels.shape, dtype=object)
    for i in range(positions.shape[0]):
        print('{} of {}'.format(i, positions.shape[0]))

        color = [0, 0, 0, 1]
        azimuth = map.get_azimuth(positions[i,:])
        if azimuth:
            color[0] = max(azimuth / 10, 0)

        # elevation = map.get_elevation(positions[i,:])
        # if elevation:
        #     color[1] = min(max(elevation / 5, 0), 1)

        print('colour {}'.format(color))
        colors[positions[i][0], positions[i][1], positions[i][2]] = color

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.set_xlabel('back')
    ax.set_ylabel('down')
    ax.set_zlabel('lateral')
    ax.set_xlim((min(positions[:,0]), max(positions[:,0])))
    ax.set_ylim((min(positions[:,1]), max(positions[:,1])))
    ax.set_zlim((min(positions[:,2]), max(positions[:,2])))
    ax.azim = -90
    ax.elev = -15
    ax.set_title('Barrels')
    plt.tight_layout()
    plt.show()


def plot(cache, map, centres):
    source_mask = cache.get_source_mask()
    source_keys = source_mask.get_key(structure_ids=None)
    source_key_volume = source_mask.map_masked_to_annotation(source_keys)

    positions = map.positions
    # positions = map.border._get_positions('SSp-bfd')
    # positions = map.border._get_positions('SSp-bfd4')

    voxels = source_key_volume < 0 #all False
    voxels[positions[:,0], positions[:,1], positions[:,2]] = True

    colors = np.empty(voxels.shape, dtype=object)
    for i in range(positions.shape[0]):
        print('{} of {}'.format(i, positions.shape[0]))
        barrel_index = get_barrel(centres, positions[i,:], map) #could do get azimuth / elevation?
        c = get_colour(barrel_index)
        print(c)
        colors[positions[i][0], positions[i][1], positions[i][2]] = c

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.set_xlabel('back')
    ax.set_ylabel('down')
    ax.set_zlabel('lateral')
    ax.set_xlim((min(positions[:,0]), max(positions[:,0])))
    ax.set_ylim((min(positions[:,1]), max(positions[:,1])))
    ax.set_zlim((min(positions[:,2]), max(positions[:,2])))
    ax.azim = -90
    ax.elev = -15
    ax.set_title('Barrels')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
    map = WhiskerMap(cache)

    # flat_border_SSp_bfd = map.border.get_flatmap_border('SSp-bfd')
    barrel_border = np.array(barrel_border)
    centres = np.array(barrel_centres)
    barrel_border[:, 0] = -barrel_border[:, 0]
    centres[:,0] = -centres[:,0]
    angle = -.55
    # border = rotate(border, angle)
    # centres = rotate(centres, angle)
    scale_factor = 18.25
    # border = scale(border, scale_factor)
    # centres = scale(centres, scale_factor)
    offset = (70.75,77.125)
    # border = shift(border, offset)
    # centres = shift(centres, offset)
    barrel_border = transform(barrel_border, angle, scale_factor, offset)
    centres = transform(centres, angle, scale_factor, offset)
    bx = [x for x in barrel_border[:, 0]]
    by = [y for y in barrel_border[:, 1]]
    bx.append(bx[0])
    by.append(by[0])
    plt.plot(bx, by, 'k')
    plt.plot(centres[:,0], centres[:,1], 'ko')

    positions = get_positions(cache, 'SSp-bfd4')
    # positions = map.border._get_positions('SSp-bfd4') # back, down, lateral
    draw_voxel_squares(positions)
    plt.xlabel('posterior')
    plt.ylabel('lateral')

    print(positions.shape)
    plt.tight_layout()
    plt.show()

    map.set_barrel_centers(centres)
    map.set_border(barrel_border)
    ae = np.array(barrel_azimith_elevation)
    map.set_azimuth_elevation(ae[:,0], ae[:,1])

    plot_ae(cache, map)
    # plot(cache, map, centres)
