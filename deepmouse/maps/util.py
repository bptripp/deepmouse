"""
Utility code for the Allen API.
"""
import os
import numpy as np
import pickle
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from mcmodels.core import VoxelModelCache


class ResultCache:
    """
    Storage of expensive computational results.
    """

    cache = {}

    @classmethod
    def get(cls, name, filepath=None, function=None):
        """
        :param name: Key for storage and retrieval of a computational result
        :param filepath: Path to pickle file for storage of result (defaults to [name].pkl)
        :param function: Code to generate result (must be a zero-arg function that returns result)
        :return: the result
        """
        if name in cls.cache:
            return cls.cache[name]

        if filepath is None:
            filepath = '{}.pkl'.format(name)

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as file:
                result = pickle.load(file)
            cls.cache[name] = result
            return result

        if function is not None:
            result = function()
            with open(filepath, 'wb') as file:
                pickle.dump(result, file)
            cls.cache[name] = result
            return result

        raise Exception('Unknown result: ' + name)


def get_ancestor_ids(structure_tree, structure_id):
    # note these are return in ascending order beginning with structure_id
    return structure_tree.get_ancestor_id_map()[structure_id]


def get_descendent_ids(structure_tree, structure_id):
    return [key for key in structure_tree.get_ancestor_id_map().keys() if
                structure_tree.structure_descends_from(key, structure_id)]


def get_child_ids(structure_tree, structure_id):
    map = structure_tree.get_ancestor_id_map()
    result = []
    for key in map.keys():
        if len(map[key]) > 1 and map[key][1] == structure_id:
            result.append(key)
    return result


def get_leaf_descendent_ids(structure_tree, structure_id):
    result = []
    for child_id in get_child_ids(structure_tree, structure_id):
        if len(get_child_ids(structure_tree, child_id)) == 0: # is leaf
            result.append(child_id)
        else:
            result.extend(get_leaf_descendent_ids(structure_tree, child_id))
    return result


def get_name(structure_tree, structure_id):
    return structure_tree.get_name_map()[structure_id]


def get_acronym(structure_tree, structure_id):
    return structure_tree.get_structures_by_id([structure_id])[0]['acronym']


def get_id(structure_tree, acronym):
    return structure_tree.get_id_acronym_map()[acronym]


def search_id_by_name(structure_tree, partial_name):
    map = structure_tree.get_name_map()
    partial_name = partial_name.lower()
    result = []
    for key in map.keys():
        if partial_name in map[key].lower():
            result.append(key)
            # print('{}: {}'.format(key, map[key]))
    return result


def search_id_by_acronym(structure_tree, partial_acronym):
    map = structure_tree.get_id_acronym_map()
    result = []
    for key in map.keys():
        if partial_acronym in key:
            result.append(map[key])
    return result


def get_voxel_model_cache():
    return VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')


def get_default_structure_tree():
    mcc = MouseConnectivityCache() # there is a resolution arg that defaults to 25
    return mcc.get_structure_tree()


def print_descriptions(structure_tree, structure_ids, verbose=False):
    if not isinstance(structure_ids, list):
        structure_ids = [structure_ids]

    for structure_id in structure_ids:
        if verbose:
            print(structure_tree.get_structures_by_id([structure_id]))
        else:
            acronym = get_acronym(structure_tree, structure_id)
            name = get_name(structure_tree, structure_id)
            print('{} ({}): {}'.format(acronym, structure_id, name))


def get_positions(cache, id):
    # similar function in map
    source_mask = cache.get_source_mask()
    source_keys = source_mask.get_key(structure_ids=None)
    structure_tree = cache.get_structure_tree()

    mask_indices = np.array(source_mask.mask.nonzero())

    positions = []
    for i in range(len(source_keys)):  # single hemisphere
        if structure_tree.structure_descends_from(source_keys[i], id):
            positions.append(mask_indices[:, i])

    return np.array(positions)


# TODO: is this redundant with map.get_positions?
def get_area_positions(cache, area):
    structure_tree = get_default_structure_tree()
    structure_id = get_id(structure_tree, area)

    source_mask = cache.get_source_mask()

    source_keys = source_mask.get_key(structure_ids=[structure_id])
    source_key_volume = source_mask.map_masked_to_annotation(source_keys)
    positions_3d = np.array(np.nonzero(source_key_volume))
    return positions_3d


if __name__ == '__main__':
    structure_tree = get_default_structure_tree()

    # print(get_id(structure_tree, 'VISp'))
    # print(get_id(structure_tree, 'VISp2/3'))
    # print(get_id(structure_tree, 'VISpm4'))

    # ids = search_id_by_name(structure_tree, 'tract')
    # for id in ids:
    #     print_descriptions(structure_tree, id)

    # ids = get_ancestor_ids(structure_tree, 245) #
    # print(ids)


    # print(search_id_by_name(structure_tree, 'cortex'))
    # vis_ids = search_id_by_acronym(structure_tree, 'VIS')
    # print(vis_ids)
    # for id in [245, 871, 967, 1009, 997]:
    #     print_descriptions(structure_tree, id)

    # print_descriptions(structure_tree, 315) #isocortex

    # for id in get_child_ids(structure_tree, 315):
    #     print_descriptions(structure_tree, id)

    chars = '123456'
    for id in get_descendent_ids(structure_tree, 315):
        if not any((c in chars) for c in get_name(structure_tree, id)):
            print_descriptions(structure_tree, id)

    # cache = get_voxel_model_cache()
    # p = get_positions(cache, 315)
    # print(p.shape)
    # p = get_positions(cache, 385)
    # print(p.shape)

    #
    # ids = get_ancestor_ids(structure_tree, 312782558)
    # print_descriptions(structure_tree, ids)
    #
    # ids = get_child_ids(structure_tree, 22)
    # # print_descriptions(structure_tree, 22)
    # print_descriptions(structure_tree, ids, verbose=True) # includes areas as well as layers
    #
    # # this just returns a list of IDs, not sure what the sets are; maybe some are layers?
    # # print(structure_tree.get_structure_sets())
    #
    # ids = get_descendent_ids(structure_tree, 22)
    # print_descriptions(structure_tree, ids)


    # VISrl1 doesn't have "vis" in name although parent does
    # VISC areas have VIS in acronym
    # get_structures_by_acronym(acronyms) #what does this do?