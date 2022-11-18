
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mcmodels.core import VoxelModelCache
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.maps.flatmap import FlatMap
from deepmouse.geodesic_flatmap import GeodesicFlatmap
from deepmouse.maps.map import right_target_indices, get_positions, right_target_indices_area
from tqdm import tqdm


cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()

def load_weights(data_folder="data_files/"):

    weights = cache.get_weights()
    nodes = cache.get_nodes()

    return weights, nodes

def get_target_cortex_keys():
   
    cortex_id = structure_tree.get_id_acronym_map()['Isocortex']

    target_mask = cache.get_target_mask()
    target_keys = target_mask.get_key(structure_ids=None) # structure ids for all voxels
    print('len target keys {}'.format(len(target_keys)))
    
    target_cortex_indices = []
    target_cortex_keys = []
    for i in range(len(target_keys)):
        if structure_tree.structure_descends_from(target_keys[i], cortex_id):
            target_cortex_indices.append(i)
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

def test_get_target_cortex_keys(target_cortex_keys, target_positions):
    
    
    random_indices = np.random.randint(0, len(target_positions),(3,))
    for idx in tqdm(random_indices):
        voxel = target_positions[idx]
        sel_target_idx = get_voxel_same_area_indices(voxel, target_positions, target_cortex_keys)
        target_key = target_cortex_keys[sel_target_idx[0]]

        area_r = right_target_indices_area(cache, target_key)
        area_positions = get_positions(cache,target_key,True)[area_r]
        
        sel_target_positions = target_positions[sel_target_idx]
        assert len(sel_target_positions) == len(area_positions)
        assert (sel_target_positions==sel_target_positions).all(axis=1).all()
        
target_cortex_keys = get_target_cortex_keys()
target_positions = get_target_positions()
test_get_target_cortex_keys(target_cortex_keys, target_positions)