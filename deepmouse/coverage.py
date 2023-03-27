import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.maps.map import get_positions
from deepmouse.geodesic_flatmap import GeodesicFlatmap, concave_hull
from deepmouse.topography import Gaussian2D, get_primary_gaussians


cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()

# source_areas = ['VISp', 'AUDp', 'SSp-bfd', 'SSP-ul']
source_areas = ['VISp', 'SSp-bfd']
test_areas = ['RSP']

propagated = []
for sa in source_areas:
    with open('generated/propagated {}'.format(sa), 'rb') as file:
        propagated.append(pickle.load(file))

flatmap = GeodesicFlatmap()


def plot_ellipse(mean, covariance, rel_weight):
    angle = np.linspace(0, 2 * np.pi, num=100)
    unit_circle = np.vstack([np.cos(angle), np.sin(angle)])
    x, y = sqrtm(covariance) @ unit_circle # https://gist.github.com/CarstenSchelp/b992645537660bda692f218b562d0712
    plt.plot(mean[0] + x, mean[1] + y, 'k', linewidth=.1 * rel_weight)


# plot_ellipse([0, 0], [[4, -1], [-1, 4]], 1)
# plot_ellipse([1, 1], [[3, 0], [0, 2]], .5)
# plt.show()


def get_boundary(area):
    gaussians = get_primary_gaussians(area)
    points = [[g.mean] for g in gaussians]
    return concave_hull(points)


def plot_coverage(area_gaussians, test_area):
    """
    :param area_gaussians: Gaussian models of input coords from one area for all cortical voxels
    :param test_area: name of single cortical area to plot coverage for
    """
    positions_3d = get_positions(cache, test_area)
    indices = [flatmap.get_voxel_index(p) for p in positions_3d] # indices of voxels in test_area

    weights = [area_gaussians[index].weight for index in indices]
    for i, index in enumerate(indices):
        g = area_gaussians[index]
        plot_ellipse(g.mean, g.covariance, g.weight/max(weights))
    plt.show()


def plot_multi_coverage(area_gaussians1, area_gaussians2, test_area):
    positions_3d = get_positions(cache, test_area)
    indices = [flatmap.get_voxel_index(p) for p in positions_3d] # indices of voxels in test_area
    weights1 = [area_gaussians1[index].weight for index in indices]
    weights2 = [area_gaussians2[index].weight for index in indices]
    max_weight = max(weights1 + weights2)

    plt.figure(figsize=(6,6))
    plt.subplot(2,2,1)
    for i, index in enumerate(indices):
        g = area_gaussians1[index]
        plot_ellipse(g.mean, g.covariance, g.weight/max_weight)
    plt.subplot(2,2,3)
    for i, index in enumerate(indices):
        g1 = area_gaussians1[index]
        g2 = area_gaussians2[index]
        mean = [g1.mean[0], g2.mean[1]]
        cov = [[g1.covariance[0][0], 0], [0, g2.covariance[1][1]]]
        plot_ellipse(mean, cov, (g1.weight*g2.weight)**.5/max_weight)
    plt.subplot(2,2,2)
    for i, index in enumerate(indices):
        g1 = area_gaussians1[index]
        g2 = area_gaussians2[index]
        mean = [g2.mean[0], g1.mean[1]]
        cov = [[g2.covariance[0][0], 0], [0, g1.covariance[1][1]]]
        plot_ellipse(mean, cov, (g1.weight*g2.weight)**.5/max_weight)
    plt.subplot(2,2,4)
    for i, index in enumerate(indices):
        g = area_gaussians2[index]
        plot_ellipse(g.mean, g.covariance, g.weight/max_weight)
    # plt.savefig('')
    plt.show()


# foo = get_boundary(source_areas[0])
# plt.plot(foo[:,0], foo[:,1])
# plt.show()

# plot_coverage(propagated[0], test_areas[0])
plot_multi_coverage(propagated[0], propagated[1], test_areas[0])

