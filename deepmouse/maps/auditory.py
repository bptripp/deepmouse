"""
Frequency maps in primary auditory cortex are shown in:

﻿J. B. Issa, B. D. Haeffele, A. Agarwal, D. E. Bergles, E. D. Young, and D. T. Yue,
“Article Multiscale Optical Ca 2 + Imaging of Tonal Organization in Mouse Auditory Cortex,”
Neuron, vol. 83, no. 4, pp. 944–959, 2014.

Tsukano, H., Horie, M., Hishida, R., Takahashi, K., Takebayashi, H., & Shibuki, K. (2016).
Quantitative map of multiple auditory cortical regions with a stereotaxic fine-scale atlas
of the mouse brain. Scientific reports, 6, 22315.

Registration with Allen Atlas isn't obvious, but both these maps show a gradient from low
to high frequencies, moving from dorsal-posterior to ventral-anterior in primary auditory
cortex, beginning at 3KHz. The endpoint isn't clear for Issa et al., but is not at the top
of the scale (96KHz). The endpoint in Tsukano et al. is 30KHz.

The approach here is is as follows:
- find the principal axis of AUDp layer 2/3
- assign frequencies 3 and 30 KHz to the dorsal-caudal and ventral-rostral ends of this axis
- assign intermediate frequencies to other L2/3 positions, using a log scale, and taking the
    distance along the scale as the distance to the low-frequency pole divided by the total
    distance to both poles
- assign preferred frequencies in other layers according to the closest point in L2/3

Compared to a linear gradient of frequency preferences, this approach emphasizes intermediate
frequencies, qualitatively consistent with Issa et al. Fig. 2D.

See also Guo et al. (2012) J Neurosci. re. clarity of topography in different layers. 
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmodels.core import VoxelModelCache
from deepmouse.maps.map import get_positions

#TODO: rather than closest 2/3 voxel, use direction in which cortex is thinnest

class FrequencyMap:
    """
    Estimates frequency map for primary auditory cortex voxels.
    """
    def __init__(self, cache=None):
        self.cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
        self._positions23 = get_positions(self.cache, 'AUDp2/3')
        self._poles = get_poles(self._positions23)
        # print(self._poles)

        # print(self._positions23)
        frequencies = [get_preferred_frequency(p, self._poles) for p in self._positions23]
        self._frequencies23 = np.array(frequencies)
        # print(self._frequencies23)

    def get_preferred_frequency(self, position):
        l23_distances = np.linalg.norm(position - self._positions23, axis=1)
        return self._frequencies23[np.argmin(l23_distances)]

    def plot(self):
        source_mask = self.cache.get_source_mask()
        source_keys = source_mask.get_key(structure_ids=None)
        source_key_volume = source_mask.map_masked_to_annotation(source_keys)

        positions = get_positions(self.cache, 'AUDp2/3')

        voxels = source_key_volume < 0 #all False
        voxels[positions[:,0], positions[:,1], positions[:,2]] = True

        colors = np.empty(voxels.shape, dtype=object)
        for i in range(positions.shape[0]):
            print('{} of {}'.format(i, positions.shape[0]))
            pf = self.get_preferred_frequency(positions[i,:])
            fraction_high = (np.log(pf) - np.log(3000)) / (np.log(30000) - np.log(3000))
            c = [fraction_high, 0, 1-fraction_high, .5]
            colors[positions[i][0], positions[i][1], positions[i][2]] = c

        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111, projection='3d')
        # ax.voxels(voxels, facecolors=[1, 0, 0, .1], edgecolor='k')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')
        ax.set_xlabel('back')
        ax.set_ylabel('down')
        ax.set_zlabel('lateral')
        ax.set_xlim((min(positions[:,0]), max(positions[:,0])))
        ax.set_ylim((min(positions[:,1]), max(positions[:,1])))
        ax.set_zlim((min(positions[:,2]), max(positions[:,2])))
        ax.azim = -90
        ax.elev = -15
        ax.set_title('AUDp2/3')

        poles = np.array(self._poles)
        plt.plot(poles[:,0], poles[:,1], poles[:,2], 'k-x')

        plt.tight_layout()
        plt.show()



def get_poles(positions23):
    """
    :param positions23: positions of AUDp2/3 voxels
    :return: (position of caudal pole, position of rostral pole)
    """
    # return ([86, 25, 104], [75, 33, 105])
    return ([86, 24, 104], [75, 33, 105])
    # U, s, Vt = np.linalg.svd(positions23, full_matrices=False)
    # pc1 = Vt[0,:] # back, down, lateral
    #
    # scalar_projections = [np.dot(p, pc1) for p in positions23]
    # min_index = np.argmin(scalar_projections)
    # max_index = np.argmax(scalar_projections)
    #
    # if positions23[min_index,0] > positions23[max_index,0]:
    #     caudal_pole = positions23[min_index]
    #     rostral_pole = positions23[max_index]
    # else:
    #     caudal_pole = positions23[max_index]
    #     rostral_pole = positions23[min_index]
    #
    # return (caudal_pole, rostral_pole)


def get_preferred_frequency(position23, poles):
    distance_to_low = np.linalg.norm(position23 - poles[0]) # caudal pole
    distance_to_high = np.linalg.norm(position23 - poles[1]) # rostral pole
    fraction = distance_to_low / (distance_to_low + distance_to_high)
    log_low = np.log(3000)
    log_high = np.log(30000)
    log_preferred = log_low + fraction*(log_high-log_low)
    # print('{} {} {}'.format(fraction, log_preferred, np.exp(log_preferred)))
    return np.exp(log_preferred)


if __name__ == '__main__':
    fm = FrequencyMap()
    fm.plot()

        # self.border = Border()
        #
        # flat_border_AUDp = self.border.get_flatmap_border('AUDp')
        # flat_centre_AUDp = find_centre(flat_border_AUDp)
        # flat_centre_VISam = find_centre(self.border.get_flatmap_border('VISam'))
        # flat_centre_VISal = find_centre(self.border.get_flatmap_border('VISal'))
        # flat_angle_VISam = get_angle(flat_centre_VISam - flat_centre_AUDp)
        # flat_angle_VISal = get_angle(flat_centre_VISal - flat_centre_AUDp)
        #
        # if plot_borders:
        #     plt.subplot(1,2,1)
        #     fam = self.border.get_flatmap_border('VISam')
        #     fal = self.border.get_flatmap_border('VISal')
        #     plt.plot(flat_border_VISp[:,0], flat_border_VISp[:,1], color='k')
        #     plt.plot(fam[:,0], fam[:,1], color='r')
        #     plt.plot(fal[:,0], fal[:,1], color='b')
        #     plt.subplot(1,2,2)
        #     gp = np.array(VISp_border)
        #     gam = np.array(VISam_border)
        #     gal = np.array(VISal_border)
        #     plt.plot(gp[:,0], gp[:,1], color='k')
        #     plt.plot(gam[:,0], gam[:,1], color='r')
        #     plt.plot(gal[:,0], gal[:,1], color='b')
        #     plt.show()
        #
        # garrett_centre_VISp = find_centre(VISp_border)
        # garrett_centre_VISam = find_centre(VISam_border)
        # garrett_centre_VISal = find_centre(VISal_border)
        # garrett_angle_VISam = get_angle(garrett_centre_VISam - garrett_centre_VISp)
        # garrett_angle_VISal = get_angle(garrett_centre_VISal - garrett_centre_VISp)
        #
        # print('flat angles: am {} al {}'.format(flat_angle_VISam, flat_angle_VISal))
        # print('garrett angles: am {} al {}'.format(garrett_angle_VISam, garrett_angle_VISal))
        #
        # angle_difference = garrett_angle_VISam + garrett_angle_VISal \
        #                    - flat_angle_VISam - flat_angle_VISal
        #
        # print('angle difference {}'.format(angle_difference))
        #
        # morph = Morph(flat_border_VISp, VISp_border, angle_difference)
        #
        # azimuth_field = Field(VISp_border, VISp_azimuth_contours, (-60, 60))
        # altitude_field = Field(VISp_border, VISp_altitude_contours, (-40, 40))
        #
        # self.azimuths = []
        # self.altitudes = []
        #
        # flatmap = FlatMap.get_instance()
        #
        # self.positions = self.border._get_positions('VISp')
        # for position in self.positions:
        #     flat_position = flatmap.get_position_2d(position)
        #     point = morph.map_to_target(flat_position)
        #     self.azimuths.append(azimuth_field.get_value(point))
        #     self.altitudes.append(altitude_field.get_value(point))

