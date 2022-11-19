
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mcmodels.core import VoxelModelCache
# from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
# from deepmouse.maps.flatmap import FlatMap
# from deepmouse.geodesic_flatmap import GeodesicFlatmap
# from deepmouse.maps.map import right_target_indices, get_positions, right_target_indices_area
from maps.util import get_voxel_model_cache, get_default_structure_tree
from maps.flatmap import FlatMap
from geodesic_flatmap import GeodesicFlatmap
from maps.map import right_target_indices, get_positions, right_target_indices_area
from tqdm import tqdm


cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()

cortical_areas = ["FRP"]
cortical_areas += ["MOp", "MOs"]
cortical_areas += [
    "SSp-n",
    "SSp-bfd",
    "SSp-ll",
    "SSp-m",
    "SSp-ul",
    "SSp-tr",
    "SSp-un",
    "SSs",
]
cortical_areas += ["GU", "VISC"]
cortical_areas += ["AUDd", "AUDp", "AUDpo", "AUDv"]
cortical_areas += [
    "VISp",
    "VISal",
    "VISam",
    "VISl",
    "VISpl",
    "VISpm",
    "VISli",
    "VISpor",
    "VISa",
    "VISrl",
]
cortical_areas += ["ACAd", "ACAv"]
cortical_areas += ["PL", "ILA"]
cortical_areas += ["ORBl", "ORBm", "ORBvl"]
cortical_areas += ["AId", "AIp", "AIv"]
cortical_areas += ["RSP", "RSPagl", "RSPd", "RSPv"]
cortical_areas += ["TEa", "PERI"]
cortical_areas += ["ECT"]


def load_weights(data_folder="data_files/"):

    weights = cache.get_weights()
    nodes = cache.get_nodes()

    return weights, nodes

def get_target_cortex_keys():
    cortex_id = structure_tree.get_id_acronym_map()['Isocortex']
    target_mask = cache.get_target_mask()
    target_keys = target_mask.get_key(structure_ids=None) # structure ids for all voxels
    print('len target keys {}'.format(len(target_keys)))
    # get the ancestor ID map
    ancestor_map = structure_tree.get_ancestor_id_map()
    target_cortex_indices = []
    target_cortex_keys = []
    for i in range(len(target_keys)):
        if structure_tree.structure_descends_from(target_keys[i], cortex_id):
            target_cortex_indices.append(i)
            # Find the parent ID for current target key
            parent_id = ancestor_map[target_keys[i]][1]
            # find the parent name for the parent ID
            parent_name = structure_tree.get_structures_by_id([parent_id])[0]['acronym']
            # if the parent for current target key is in cortical areas, use parent id
            if parent_name in cortical_areas:
                target_cortex_keys.append(parent_id)
            else:
                target_cortex_keys.append(target_keys[i])
    
    target_cortex_indices = np.array(target_cortex_indices)
    target_cortex_keys = np.array(target_cortex_keys)

    r = right_target_indices(cache)
    
    right_target_cortex_keys = target_cortex_keys[r]
    return right_target_cortex_keys
    

def get_voxel_same_area_indices(voxel, target_positions, target_cortex_keys):
    target_key = target_cortex_keys[np.where((voxel==target_positions).all(axis=1))[0]]
    return np.where(target_cortex_keys == target_key)[0]


def get_target_positions():
    cortex_id = structure_tree.get_id_acronym_map()['Isocortex']
    target_mask = cache.get_target_mask()
    target_keys = target_mask.get_key(structure_ids=None)
    target_cortex_indices = []
    for i in range(len(target_keys)):
        if structure_tree.structure_descends_from(target_keys[i], cortex_id):
            target_cortex_indices.append(i)

    #right_target_cortex_indices will not be same as source_cortex_indices but positions in same order
    target_cortex_indices = np.array(target_cortex_indices)
    
    r = right_target_indices(cache)
    right_target_cortex_indices = target_cortex_indices[r]
    target_positions = get_positions(cache,"root",True)[right_target_cortex_indices]
    return target_positions

def main():
    target_cortex_keys = get_target_cortex_keys()
    target_positions = get_target_positions()
    # for voxel in tqdm(target_positions[:10]):
    #     get_voxel_same_area_indices(voxel, target_positions, target_cortex_keys)
    # test_get_target_cortex_keys(target_cortex_keys, target_positions)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # Add x, y gridlines
    ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
    # Creating color map
    my_cmap = plt.get_cmap("plasma")
    areas = ["VISp","FRP","AUDp","SSp-bfd","MOp"]
    positions = []
    for area in areas:
        area_r = right_target_indices_area(cache, area)
        visp_positions = get_positions(cache,area,True)[area_r]
        indices = get_voxel_same_area_indices(visp_positions[0], target_positions, target_cortex_keys)
        target_visp_positions = target_positions[indices]
        # Creating plot
        print(area, len(target_visp_positions))
        sctt = ax.scatter3D(
            target_visp_positions[:, 0],
            target_visp_positions[:, 2],
            target_visp_positions[:, 1],
            alpha=0.1,
            # cmap=my_cmap,
            s=2,
            marker="o",
        )
        positions.append(np.mean(target_visp_positions, axis=0))

    for pos, area in zip(positions, areas):
        ax.text(pos[0], pos[2], pos[1], f"{area}", fontsize=7)

    from textwrap import wrap
    areas_label = ",".join(areas)
    title = ax.set_title(f" target position for {areas_label}")
    ax.set_xlim([0, 110])
    ax.set_zlim([0, 60])
    ax.axes.yaxis.set_ticklabels([])
    ax.invert_zaxis()
    ax.set_yticklabels([])
    # fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(azim=-83, elev=1)
    # plt.tight_layout()
    plt.savefig("{}/target_pos_{}.png".format(".", areas_label))
    plt.close()