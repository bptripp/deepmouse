import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from deepmouse.binzegger import Binzegger04

"""
Mouse connectivity data. Major sources include:

Ero, C., Gewaltig, M. O., Keller, D., & Markram, H. (2019). A Cell Atlas for the Mouse Brain.
Frontiers in Neuroinformatics, 13, 7.

Thomson, A. M., & Lamy, C. (2007). Functional maps of neocortical local circuitry.
Frontiers in Neuroscience, 1, 2.

Harris, J. A., Mihalas, S., Hirokawa, K. E., Whitesell, J. D., Knox, J., Bernard, A., ... & Feng, D. 
(2018). The organization of intracortical connections by layer and cell class in the mouse brain.
BioRxiv, 292961.

"""

data_folder = 'data_files'


class Data:
    def __init__(self):
        self.e18 = Ero2018()
        self.p11 = Perin11()
        self.t07 = ThomsonLamy07()

    def get_areas(self):
        return ['LGNd', 'LGNv', 'VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam']

    def get_layers(self):
        return ['2/3', '4', '5', '6']

    def get_hierarchical_level(self, area):
        hierarchy = {
            'LGNd': 0, 'LGNv': 0,
            'VISp': 1,
            'VISpl': 2, 'VISl': 2, 'VISrl': 2, 'VISpm': 2,
            'VISli': 3, 'VISa': 3, 'VISal': 3, 'VISam': 3
        }
        return hierarchy[area]

    def get_num_neurons(self, area, layer):
        self.e18.get_n_excitatory(area, layer)

    def get_input_field_size(self, source_area, source_layer, target_area, target_layer):
        # units are micrometers of cortex
        pass

    def get_extrinsic_inputs(self, target_area, target_layer):
        return 200 #TODO

    def get_connection_probability_peak(self, source_layer, target_layer):
        self.t07.hit_rate(source_layer, target_layer)

    def get_connection_probability_width(self, source_layer, target_layer):
        return self.p11.width_micrometers


class Ero2018:
    """
    Data from supplementary material of [1]. We load names of regions and numbers of
    excitatory neurons.
    """

    def __init__(self):
        self.regions = []
        self.excitatory = []
        with open(data_folder + '/Data_Sheet_1_A Cell Atlas for the Mouse Brain.CSV') as csvfile:
            r = csv.reader(csvfile)
            header_line = True
            for row in r:
                if header_line:
                    header_line = False
                else:
                    self.regions.append(row[0])
                    self.excitatory.append(row[4])

    def get_n_excitatory(self, area, layer=None):
        area_map = {
            'LGNd': 'Dorsal part of the lateral geniculate complex',
            'LGNv': 'Ventral part of the lateral geniculate complex',
            'VISal': 'Anterolateral visual area',
            'VISam': 'Anteromedial visual area',
            'VISl': 'Lateral visual area',
            'VISp': 'Primary visual area',
            'VISpl': 'Posterolateral visual area',
            'VISpm': 'posteromedial visual area'
        }

        if layer is None:
            index = self.regions.index(area_map[area])
            result = np.int(self.excitatory[index])
        elif layer == '6':
            index_a = self.regions.index('{} layer 6a'.format(area_map[area]))
            index_b = self.regions.index('{} layer 6a'.format(area_map[area]))
            result = np.int(self.excitatory[index_a]) + np.int(self.excitatory[index_b])
        else:
            index = self.regions.index('{} layer {}'.format(area_map[area], layer))
            result = np.int(self.excitatory[index])

        return result


class ThomsonLamy07:
    """
    Data on "hit rates" of various interlaminar connections, i.e. fraction of
    tested pairs of neurons that had a functional connection, reviewed in:

    [1] Thomson and Lamy, Functional maps of neocortical local circuitry,
    Front. Neurosci., vol. 1, no. 1, pp. 19 42, 2007.

    Hit rates and # pairs are from their Table 1. Where available we take adult rat data,
    and ignore e.g. juvenile rat or cat data.
    """
    def __init__(self):
        # Note much smaller hit rate for L3P-> L5 small adapt, but no count given

        # each tuple: source layer, target layer, hit rate, # pairs tested
        self.data = [
            ('4', '2/3', .28, 7), # Thomson et al. (2002)
            ('4', '2/3', .3, 5), # Bannister & Thomson (2007)
            ('2/3', '5', .55, 16), # Thomson et al. (2002)
            ('2/3', '5', .25, 17), # Thomson & Bannister (1998)
            ('4', '5', .1, 12), # Feldmeyer et al. (2005) (this is with P17-23 rats)
            ('5', '5', .014, 25), # Thomson et al. (1993)
            ('5', '5', .09, 15) # Thomson et al. (2002)
        ]
        self.b04 = Binzegger04()
        self.l6_coeff = self._fit_hit_rate_to_synapses_per_neuron()

    def hit_rate(self, source_layer, target_layer):
        if target_layer == '6':
            # minimal hit rate data; estimate using Binzegger
            return self.l6_coeff * self.b04.synapses_per_neuron(source_layer, '6')
        else:
            hit_rates = []
            counts = []
            for d in self.data:
                if d[0] == source_layer and d[1] == target_layer:
                    hit_rates.append(d[2])
                    counts.append(d[3])

            if len(hit_rates) > 0:
                return np.average(hit_rates, weights=counts)
            else:
                raise Exception('No data found for connection {}->{}'.format(source_layer, target_layer))

    def _fit_hit_rate_to_percent_evoked(self, plot=False):
        z16 = Zarrinpar16()

        hit_rate = []
        percent_evoked = []
        for source_layer in ['2/3', '4', '5']:
            hit_rate.append(self.hit_rate(source_layer, '5'))
            percent_evoked.append(z16.percent_evoked_input(source_layer))

        p = np.polyfit(percent_evoked, hit_rate, 1)

        if plot:
            plt.scatter(percent_evoked, hit_rate, color='b')
            fit = p[1] + p[0] * np.array(percent_evoked)
            plt.plot(percent_evoked, fit)
            plt.xlabel('Percent evoked input (Zarrinpar & Callaway)')
            plt.ylabel('Hit rate (Thomson & Lamy)')
            # plt.show()

        return p

    def _fit_hit_rate_to_synapses_per_neuron(self, plot=False):
        hit_rate = []
        synapses_per_neuron = []
        for source_layer in ['2/3', '4', '5']:
            hit_rate.append(self.hit_rate(source_layer, '5'))
            synapses_per_neuron.append(self.b04.synapses_per_neuron(source_layer, '5'))

        # p = np.polyfit(synapses_per_neuron, hit_rate, 1)
        A = np.array(synapses_per_neuron)[:,np.newaxis]
        x, residuals, rank, s = np.linalg.lstsq(A, hit_rate, rcond=None)

        if plot:
            synapses_per_neuron.append(0)
            hit_rate.append(0)
            plt.scatter(synapses_per_neuron, hit_rate, color='b')
            # fit = p[1] + p[0] * np.array(synapses_per_neuron)
            fit = x * np.array(synapses_per_neuron)
            plt.plot(synapses_per_neuron, fit)
            plt.xlabel('Synapses per neuron (Binzegger et al.)')
            plt.ylabel('Hit rate (Thomson & Lamy)')
            # plt.show()

        return x[0]


class Zarrinpar06:
    """
    Data on input laminar input to L6 via photostimulation, from:

    Zarrinpar and Callaway, "Local connections to specific types of Layer 6 neurons in
    the rat visual cortex" J Neurophysiol 95, pp.1751-1761, 2006.
    """

    def __init__(self):
        self.data = [1.9980019980019974,
            0,
            0,
            1.9980019980019974,
            0,
            4.955044955044986,
            1.8381618381618292,
            2.7172827172827265,
            10.069930069930063,
            39.64035964035965,
            20.37962037962037,
            36.28371628371629,
            54.5854145854146,
            76.80319680319681,
            49.23076923076924
        ]

    def percent_evoked_input(self, layer):
        # we ignore interneurons and average over L5A and L5B
        indices = {'2/3': [1, 2], '4': [4, 5], '5': [7, 8, 10, 11], '6': [13, 14]}
        percents = [self.data[index] for index in indices[layer]]
        return np.mean(percents)


class Zarrinpar16:
    """
    Data on input laminar input to L5 via photostimulation, from:

    Zarrinpar and Callaway, "Local connections to specific types of Layer 6 neurons in
    the rat visual cortex" J Neurophysiol 95, pp.1751-1761, 2006.
    """

    def __init__(self):
        self.data = [
            17.446351931330472,
            12.746781115879813,
            14.098712446351932,
            39.14163090128755,
            27.55364806866951,
            27.10300429184549,
            35.08583690987125,
            46.738197424892704,
            48.798283261802574,
            9.914163090128756,
            14.549356223175954,
            12.038626609442042
        ]

    def percent_evoked_input(self, layer):
        indices = {'2/3': [0, 1, 2], '4': [3, 4, 5], '5': [6, 7, 8], '6': [9, 10, 11]}
        percents = [self.data[index] for index in indices[layer]]
        return np.mean(percents)



class Perin11:
    """
    This class fits a Gaussian function to the connection probability vs. inter-somatic
    distance among pairs of thick-tufted L5 pyramids in P14-16 Wistar rats, from Fig. 1 of [1].

    In the source figure, I would expect "overall" to be the sum of reciprical
    and non-reciprocal, but it isn't. It doesn't look like this much affects the spatial
    profile though, just the peak (which we don't use).

    [1] Perin, Berger, and Markram, "A synaptic organizing principle for cortical neuronal
    groups", PNAS 109(13). pp. 5419-24, 2011.
    """

    def __init__(self):
        connection_probability_vs_distance = [
                [17.441860465116307, 0.21723833429098494],
                [52.79069767441864, 0.1676015362748359],
                [87.44186046511628, 0.14761544742492516],
                [122.5581395348837, 0.12294674448846282],
                [157.67441860465118, 0.09515710527111632],
                [192.55813953488376, 0.10208848701121961],
                [227.44186046511635, 0.06337617564339071],
                [262.5581395348837, 0.03480630235582299],
                [297.44186046511635, 0.07021622765899538]]

        def gaussian(x, peak, sigma):
            return peak * np.exp(-x ** 2 / 2 / sigma ** 2)

        cp = np.array(connection_probability_vs_distance)
        popt, pcov = curve_fit(gaussian, cp[:,0], cp[:,1], p0=(.2, 150))
        self.width_micrometers = popt[1]


if __name__ == '__main__':
    p11 = Perin11()
    print(p11.width_micrometers)
    # cp = np.array(connection_probability_vs_distance)
    # popt, pcov = curve_fit(gaussian, cp[:,0], cp[:,1], p0=(.2, 150))
    #
    # plt.plot(cp[:,0], cp[:,1])
    # plt.plot(cp[:,0], gaussian(cp[:,0], popt[0], popt[1]), 'k')
    # plt.show()
    # print(popt)

    # ero = Ero2018()
    # print(ero.get_n_excitatory('LGNd'))
    # print(ero.get_n_excitatory('LGNv'))
    # print(ero.get_n_excitatory('VISp', '2/3'))
    # print(ero.get_n_excitatory('VISpm', '6'))

    tl = ThomsonLamy07()
    print(tl.hit_rate('4', '2/3'))
    print(tl.hit_rate('4', '5'))
    print(tl.hit_rate('2/3', '5'))
    print(tl.hit_rate('2/3', '6'))
    print(tl.hit_rate('4', '6'))
    print(tl.hit_rate('5', '6'))

    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    tl._fit_hit_rate_to_percent_evoked(plot=True)
    plt.subplot(1,2,2)
    tl._fit_hit_rate_to_synapses_per_neuron(plot=True)
    plt.tight_layout()
    plt.show()
