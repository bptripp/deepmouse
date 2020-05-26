import networkx as nx
import matplotlib.pyplot as plt


class Population:
    def __init__(self, name, n, e, eig):
        """
        :param name:
        :param n: # of neurons (estimated from density and cortical area from Markov)
        :param e: # of extrinsic inputs per neuron (typical values by cortical layer)
        :param eig: eigenvalues of population response to ?? stimuli
        """
        self.name = name
        self.n = n
        self.e = e
        self.eig = eig

    def get_description(self):
        return '{} (#neurons={}; in-degree={}; RF-width={})'.format(self.name, self.n, self.e, self.w)

    def is_input(self):
        """
        :return: True if this population is an input to the model (this is true of the # extrinsic inputs
            per neuron is 0); False otherwise
        """
        return not self.e


#TODO: do I really need subclasses or can I have f and b, possibly None, f ignored if b not None?
class Projection:
    def __init__(self, origin, termination):
        """
        :param origin: presynaptic Population
        :param termination: postsynaptic Population
        """
        self.origin = origin
        self.termination = termination

    def get_description(self):
        return '{}->{}'.format(self.origin.name, self.termination.name)


class InterAreaProjection(Projection):
    def __init__(self, origin, termination, mean_density):
        """
        :param origin: presynaptic Population
        :param termination: postsynaptic Population
        :param mean_density: mean (across target-area voxels) of weight of connection,
            as a function of visual-space offset
        """
        Projection.__init__(self, origin, termination)
        self.mean_density = mean_density

    def get_description(self):
        return '{} (FLNe={})'.format(super().get_description(), self.f)


class InterLaminarProjection(Projection):
    def __init__(self, origin, termination, hit_rate_peak, hit_rate_width):
        """
        :param origin: presynaptic Population
        :param termination: postsynaptic Population
        :param hit_rate_peak: fraction of tested pairs with with a functional connection,
            among pairs with little horizontal offset between their cell bodies
        :param hit_rate_width: width of Gaussian approximation of decline in hit rate as a
            function of increasing horizontal distance
        """
        Projection.__init__(self, origin, termination)
        self.hit_rate_peak = hit_rate_peak
        self.hit_rate_width = hit_rate_width

    def get_description(self):
        return '{} (hit rate peak={} width={})'.format(
            Projection.get_description(self), self.hit_rate_peak, self.hit_rate_width)


class System:
    def __init__(self, min_f=1e-6):
        self.min_f = min_f
        self.input_name = 'INPUT'
        self.populations = []
        self.projections = []

    def add_input(self, n, w):
        """
        Adds a special population that represents the network input. If a parameter value is
        unknown, it should be given as None.

        :param n: number of units
        :param w: width of an image pixel in degrees visual angle
        :param name (optional): Defaults to 'INPUT'
        """
        self.populations.append(Population(self.input_name, n, None, w))

    def add(self, name, n, e, w):
        if self.find_population(name) is not None:
            raise Exception(name + ' already exists in network')

        self.populations.append(Population(name, n, e, w))

    def connect_areas(self, origin_name, termination_name, f):
        origin = self.find_population(origin_name)
        termination = self.find_population(termination_name)

        if origin is None:
            raise Exception(origin_name + ' is not in the system')
        if termination is None:
            raise Exception(termination_name + ' is not in the system')

        if f >= self.min_f:
            self.projections.append(InterAreaProjection(origin, termination, f))
        else:
            print('Omitting connection {}->{} with f={}'.format(origin_name, termination_name, f))

    def connect_layers(self, origin_name, termination_name, b):
        origin = self.find_population(origin_name)
        termination = self.find_population(termination_name)

        if origin is None:
            raise Exception(origin_name + ' is not in the system')
        if termination is None:
            raise Exception(termination_name + ' is not in the system')

        self.projections.append(InterLaminarProjection(origin, termination, b))

    def find_population(self, name):
        assert isinstance(name, str)
        result = None
        for population in self.populations:
            if population.name == name:
                result = population
                break
        return result

    def find_population_index(self, name):
        assert isinstance(name, str)
        result = None
        for i in range(len(self.populations)):
            if self.populations[i].name == name:
                result = i
                break
        return result

    def find_projection(self, origin_name, termination_name):
        assert isinstance(origin_name, str)
        assert isinstance(termination_name, str)
        result = None
        for projection in self.projections:
            if projection.origin.name == origin_name and projection.termination.name == termination_name:
                result = projection
                break
        return result

    def find_projection_index(self, origin_name, termination_name):
        assert isinstance(termination_name, str)
        for i in range(len(self.projections)):
            projection = self.projections[i]
            if projection.origin.name == origin_name and projection.termination.name == termination_name:
                result = i
                break
        return result

    def find_pre(self, termination_name):
        assert isinstance(termination_name, str)

        result = []
        for projection in self.projections:
            if projection.termination.name == termination_name:
                result.append(projection.origin)
        return result

    def normalize_FLNe(self):
        """
        The fraction of extrinsic labelled neurons per source area is determined from tract-tracing
        data. However, if a System does not contain all connections in the brain, the sum of these
        fractions will be <1. This method rescales the fractions from the literature to fractions
        within the model.
        """
        for population in self.populations:
            total_FLNe = 0
            for pre in self.find_pre(population.name):
                projection = self.find_projection(pre.name, population.name)
                if isinstance(projection, InterAreaProjection):
                    total_FLNe += projection.f

            for pre in self.find_pre(population.name):
                projection = self.find_projection(pre.name, population.name)
                if isinstance(projection, InterAreaProjection):
                    projection.f = projection.f / total_FLNe

            # Here we could reduce # extrinsic inputs to account for ignored FLNe, like this:
            # population.e = population.e * total_FLNe. However, the ignored FLNe are mostly
            # due to omission of feedback and lateral connections rather than areas outside the
            # model, and most of these are not onto L4, which is the layer getting extrinsic
            # input in the feedforward model. Ideally we would account for L4 inputs due to
            # lateral connections and areas outside the model, but we ignore these for
            # simplicity.

    def prune_FLNe(self, min_fraction):
        """
        Removes projections that have FLNe less than min_fraction.
        :param min_fraction: Minimum FLNe of projections to keep in the model.
        """

        def keep(projection):
            if isinstance(projection, InterAreaProjection) and projection.f < min_fraction:
                return False
            else:
                return True

        self.projections = [p for p in self.projections if keep(p)]

    def make_graph(self):
        graph = nx.DiGraph()

        for population in self.populations:
            graph.add_node(population.name)

        for projection in self.projections:
            graph.add_edge(projection.origin.name, projection.termination.name)

        return graph

    def print_description(self):
        for population in self.populations:
            print(population.get_description())

        for projection in self.projections:
            print(projection.get_description())

    def check_connected(self, input_indices=[0]):
        """
        Checks that all populations in the system, except for identified input populations, have at least one input.
        """
        for i in range(len(self.populations)):
            pop = self.populations[i]
            # print('{}: {}'.format(pop.name, [pre.name for pre in self.find_pre(pop.name)]))
            if i not in input_indices:
                assert self.find_pre(pop.name), '{} has no inputs'.format(pop.name)

    def merge_populations(self, to_keep, to_merge):
        """
        Combines two populations, resulting in a population that has all the connections of both.

        :param to_keep: Name of merged population to keep
        :param to_merge: Name of merged population to remove after the merge
        """
        # this could create redundant self-projections, but at the moment the code base doesn't do self-projections
        # TODO: weighted average of layer properties
        # TODO: weighted average of connection properties where connections overlap

        keep_pop = self.find_population(to_keep)
        merge_pop = self.find_population(to_merge)

        projections_to_drop = []

        for projection in self.projections:
            if projection.termination.name == to_merge:
                if projection.origin.name == to_keep: # don't need projection between merged populations
                    projections_to_drop.append(projection)
                elif self.find_projection(projection.origin.name, keep_pop.name): # new projection already exists
                    projections_to_drop.append(projection)
                else:
                    projection.termination = keep_pop

            if projection.origin.name == to_merge:
                if projection.termination.name == to_keep:
                    projections_to_drop.append(projection)
                elif self.find_projection(keep_pop.name, projection.termination.name):
                    projections_to_drop.append(projection)
                else:
                    projection.origin = keep_pop

        for projection in projections_to_drop:
            self.projections.remove(projection)

        self.populations.remove(merge_pop)


def get_example_system():
    result = System()
    result.add_input(250000, .02)
    result.add('V1', 10000000, 2000, .1)
    result.add('V2', 10000000, 2000, .2)
    result.add('V4', 5000000, 2000, .4)
    result.connect_areas('INPUT', 'V1', 1.)
    result.connect_areas('V1', 'V2', 1.)
    result.connect_areas('V1', 'V4', .5)
    result.connect_areas('V2', 'V4', .5)
    return result

def get_example_small():
    result = System()
    result.add_input(750000, .02)
    result.add('V1_4', 53000000, 500, .09)
    result.add('V1_23', 53000000, 1000, .1)
    result.add('V1_5', 27000000, 3000, .11)
    result.add('V2_4', 33000000, 500, .19)
    result.add('V2_23', 33000000, 1000, .2)
    result.add('V2_5', 17000000, 3000, .21)
    result.add('V4_4', 17000000, 500, .39)
    result.add('V4_23', 17000000, 1000, .4)
    result.add('V4_5', 8000000, 3000, .41)

    result.connect_areas('INPUT', 'V1_4', 1.)

    result.connect_layers('V1_4', 'V1_23', 800.)
    result.connect_layers('V1_23', 'V1_5', 3000.)

    result.connect_areas('V1_5', 'V2_4', 1.)

    result.connect_layers('V2_4', 'V2_23', 800.)
    result.connect_layers('V2_23', 'V2_5', 3000.)

    result.connect_areas('V1_5', 'V4_4', .15)
    result.connect_areas('V2_5', 'V4_4', .85)

    result.connect_layers('V4_4', 'V4_23', 800.)
    result.connect_layers('V4_23', 'V4_5', 3000.)

    return result

def get_example_medium():
    # This example was written before the code distinguished interarea and interlaminar
    # connections. Interarea connections are used throughout (even between layers) to
    # preserve it as-is.

    result = System()
    result.add_input(750000, .02)
    result.add('LGNparvo', 2000000, 1000, .04)
    result.add('V1_4', 53000000, 500, .1)
    result.add('V1_23', 53000000, 1000, .13)
    result.add('V2_4', 33000000, 500, .2)
    result.add('V2_23', 33000000, 1000, .26)
    result.add('V4_4', 17000000, 500, .4)
    result.add('V4_23', 17000000, 1000, .5)
    result.add('MT_4', 4800000, 500, 1.)
    result.add('MT_23', 4800000, 1000, 1.1)
    result.add('VOT_4', 6000000, 500, 1.4)
    result.add('VOT_23', 6000000, 1000, 1.5)
    result.add('PITd_4', 5700000, 500, 3.)
    result.add('PITd_23', 5700000, 1000, 4.)
    result.add('DP_4', 17000000, 500, 1.7)
    result.add('DP_23', 17000000, 1000, 1.8)

    # input
    result.connect_areas('INPUT', 'LGNparvo', 1.)
    result.connect_areas('LGNparvo', 'V1_4', 1.)

    # laminar connections
    result.connect_areas('V1_4', 'V1_23', 1.)
    result.connect_areas('V2_4', 'V2_23', 1.)
    result.connect_areas('V4_4', 'V4_23', 1.)
    result.connect_areas('MT_4', 'MT_23', 1.)
    result.connect_areas('VOT_4', 'VOT_23', 1.)
    result.connect_areas('PITd_4', 'PITd_23', 1.)
    result.connect_areas('DP_4', 'DP_23', 1.)

    # feedforward inter-areal connections
    result.connect_areas('V1_23', 'V2_4', 1.)
    result.connect_areas('V1_23', 'V4_4', 0.0307)
    result.connect_areas('V1_23', 'MT_4', 0.0235)
    result.connect_areas('V2_23', 'V4_4', 0.9693)
    result.connect_areas('V2_23', 'MT_4', 0.2346)
    result.connect_areas('V2_23', 'PITd_4', 0.0026)
    result.connect_areas('V2_23', 'DP_4', 0.2400)
    result.connect_areas('V4_23', 'MT_4', 0.7419)
    result.connect_areas('V4_23', 'PITd_4', 0.2393)
    result.connect_areas('V4_23', 'DP_4', 0.7591)
    result.connect_areas('VOT_23', 'PITd_4', 0.7569)
    result.connect_areas('VOT_23', 'DP_4', 0.0008)
    result.connect_areas('MT_23', 'PITd_4', 0.0004)
    result.connect_areas('DP_23', 'PITd_4', 0.0009)
    result.connect_areas('V2_23', 'VOT_4', 0.0909)
    result.connect_areas('V4_23', 'VOT_4', 0.9091)

    return result


def get_layout(sys):
    areas = {}
    areas['INPUT'] = [.075, .5]
    areas['LGN'] = [.17, .5]
    areas['V1'] = [.15, .5]
    areas['V2'] = [.225, .6]
    areas['V3'] = [.3, .7]
    areas['V3A'] = [.35, .7]
    areas['V4'] = [.35, .4]
    areas['TEO'] = [.5, .23]
    areas['MT'] = [.425, .55]
    areas['MST'] = [.5, .55]
    areas['VIP'] = [.7, .7]
    areas['LIP'] = [.65, .65]
    areas['TEpd'] = [.6, .25]
    areas['DP'] = [.4, .5]

    offsets = {}
    offsets['4'] = 0
    offsets['4Cbeta'] = 0
    offsets['23'] = .09
    offsets['2/3'] = .09
    offsets['5'] = -.09

    result = {}
    for pop in sys.populations:
        name = pop.name.split('_')
        position = areas[name[0]].copy()
        if len(name) > 1:
            # print(offsets[name[1]])
            position[1] = position[1] + offsets[name[1]]
        result[pop.name] = position

    print(result)
    return result


if __name__ == '__main__':
    sys = get_example_medium()

    foo = get_layout(sys)
    print(foo)

    graph = sys.make_graph()
    # print(type(nx.drawing.layout.random_layout(graph)))
    #TODO: these layouts all look awful; should use flat-map cortical positions
    nx.draw_networkx(graph, pos=get_layout(sys), arrows=True, font_size=10, node_size=1200, node_color='white')
    # nx.draw_networkx(graph, pos=nx.spring_layout(graph), arrows=True, font_size=10, node_size=1200, node_color='white')
    # nx.draw_networkx(graph, pos=nx.drawing.layout.fruchterman_reingold_layout(graph), arrows=True, font_size=10, node_size=1200, node_color='white')
    plt.show()

    # from calc.conversion import make_net_from_system
    # net = make_net_from_system(sys)
    # net.print()