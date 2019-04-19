import numpy as np

data_folder = 'data_files'

class Binzegger04:
    """
    Geometric connectivity data for cat V1 [1]. We use this only as a correlate of
    hit rates, to extrapolate hit rate data to other connections.

    [1] Binzegger, T., Douglas, R. J., & Martin, K. A. (2004). A quantitative map of the
    circuit of cat primary visual cortex. Journal of Neuroscience, 24(39), 8441-8453.
    """

    def __init__(self):
        # This is per hemisphere, extracted from Fig 6A via WebPlotDigitizer
        self.BDM04_N_per_type = {
            'sp1': 25050, 'sm1': 480962,
            'p2/3': 8252505, 'b2/3': 964930, 'db2/3': 572144, 'axo2/3': 88176, 'sm2/3': 691383,
            'ss4(L4)': 2900802, 'ss4(L2/3)': 2900802, 'p4': 2900802, 'b4': 1694389, 'sm4': 480962,
            'p5(L2/3)': 1512024, 'p5(L5/6)': 389780, 'b5': 179359, 'sm5': 242485,
            'p6(L4)': 4296593, 'p6(L5/6)': 1420842, 'sm6': 1175351,
            'X/Y': 361723
        }

        self.BDM04_targets = [
            'sp1', 'sm1',
            'p2/3', 'b2/3', 'db2/3', 'axo2/3', 'sm2/3',
            'ss4(L4)', 'ss4(L2/3)', 'p4', 'b4', 'sm4',
            'p5(L2/3)', 'p5(L5/6)', 'b5', 'sm5',
            'p6(L4)', 'p6(L5/6)', 'sm6'
        ]

        self.BDM04_sources = [
            'p2/3', 'b2/3', 'db2/3', 'axo2/3',
            'ss4(L4)', 'ss4(L2/3)', 'p4', 'b4',
            'p5(L2/3)', 'p5(L5/6)', 'b5',
            'p6(L4)', 'p6(L5/6)',
            'X/Y', 'as', 'sy'
        ]

        self.BDM04_excitatory_types = {
            '1': ['sp1'],
            '2/3': ['p2/3'],
            '4': ['ss4(L4)', 'ss4(L2/3)', 'p4'],
            '5': ['p5(L2/3)', 'p5(L5/6)'],
            '6': ['p6(L4)', 'p6(L5/6)'],
            'thalamocortical': ['X/Y'],
            'extrinsic': ['as'],
            'interlaminar': ['sp1', 'p2/3', 'ss4(L4)', 'ss4(L2/3)', 'p4', 'p5(L2/3)', 'p5(L5/6)', 'p6(L4)', 'p6(L5/6)']
        }

    def _get_synapses_per_layer_cat_V1(self, layer):
        with open(data_folder + '/BDM04-Supplementary.txt') as file:
            found_layer = False
            table = []
            while True:
                line = file.readline()

                if not line:
                    break
                if found_layer and len(line.strip()) == 0:
                    break

                if not line.startswith('#'): # comment
                    if found_layer:
                        n_cols = 16
                        items = line.split()
                        row = np.zeros(n_cols)
                        assert len(items) == n_cols+1 or len(items) == 1 # expected # columns or empty
                        for i in range(1, len(items)):  # skip row header
                            row[i-1] = float(items[i].replace('-', '0'))
                        table.append(row)

                    if line.startswith('L{}'.format(layer)):
                        found_layer = True

            assert len(table) == 19 # expected # of rows
            return np.array(table)


    def synapses_per_neuron(self, source_layer, target_layer):
        n_source_types = 16
        n_target_types = 19

        # find table of synapses between types summed across all layers
        totals_across_layers = np.zeros((n_target_types, n_source_types))
        layers = ['1', '2/3', '4', '5', '6']
        for layer in layers:
            totals_across_layers = totals_across_layers + self._get_synapses_per_layer_cat_V1(layer)

        # sum over sources and weighted average over targets ...
        source_types = self.BDM04_excitatory_types[source_layer] # cell types in source layer (regardless of where synapses are)
        interlaminar_source_types = self.BDM04_excitatory_types['interlaminar']
        target_types = self.BDM04_excitatory_types[target_layer]

        total_inputs_from_source = np.zeros(n_target_types)
        total_inputs = np.zeros(n_target_types)
        for i in range(n_source_types):
            if self.BDM04_sources[i] in source_types:
                total_inputs_from_source = total_inputs_from_source + totals_across_layers[:,i]
            if self.BDM04_sources[i] in interlaminar_source_types:
                total_inputs = total_inputs + totals_across_layers[:,i]

        weighted_sum_from_source = 0.
        weighted_sum = 0.
        total_weight = 0.
        for i in range(n_target_types):
            if self.BDM04_targets[i] in target_types:
                n = self.BDM04_N_per_type[self.BDM04_targets[i]]
                weighted_sum_from_source += n * total_inputs_from_source[i]
                weighted_sum += n * total_inputs[i]
                total_weight += n

        in_degree_from_source = weighted_sum_from_source / total_weight

        return in_degree_from_source


if __name__ == '__main__':
    b04 = Binzegger04()
    print(b04.synapses_per_neuron('4', '2/3'))
