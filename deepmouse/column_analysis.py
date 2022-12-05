import numpy as np
import os
import pickle
from tqdm import tqdm

from argparse import ArgumentParser
from maps.util import get_voxel_model_cache, get_default_structure_tree, get_id
from maps.map import right_target_indices, get_positions
from geodesic_flatmap import GeodesicFlatmap
from find_voxel_cortical_area import get_target_cortex_keys, get_voxel_same_area_indices
from horizontal_distance import shortest_distance, surface_to_surface_streamline
from streamlines import CompositeInterpolator
# from topography import get_primary_gaussians, propagate_gaussians_through_isocortex, GaussianMixture2D, Gaussian2D
from topography import Gaussian2D
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--radius_multiplier","-r",default=1,type=int)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    source_areas = [
        "VISp",
        "AUDp",
        "SSp-bfd",        
        "SSp-ul",
        "SSp-m",
        "SSp-n"
    ]

    il_stds = np.array([
        [142.5,    31.67,   63.33],
        [139.33,   114,     88.67],
        [95,       63.33,   133]
    ])

    avg_std = np.mean(il_stds)

    cache = get_voxel_model_cache()
    structure_tree = get_default_structure_tree()

    # CORRECT METHOD
    positions_3d = get_positions(cache,"Isocortex")

    # Find the target indices of the right isocortex
    cortex_id = structure_tree.get_id_acronym_map()['Isocortex']
    target_mask = cache.get_target_mask()
    target_keys = target_mask.get_key(structure_ids=None)
    target_cortex_indices = []
    for i in range(len(target_keys)):
        if structure_tree.structure_descends_from(target_keys[i], cortex_id):
            target_cortex_indices.append(i)


    # Find the target positions of the right isocortex
    target_cortex_indices = np.array(target_cortex_indices)
    r = right_target_indices(cache)
    right_target_cortex_indices = target_cortex_indices[r]
    target_positions = get_positions(cache,"root",True)[right_target_cortex_indices]
    
    with open('interpolator.pkl', 'rb') as file:
        ci = pickle.load(file)
    
    test_areas = ["ILA","MOp"]
    for test_area in test_areas:
    
        print(f"Analyzing {test_area}...")
        cortex_id = structure_tree.get_id_acronym_map()[test_area]
        target_cortex_keys = get_target_cortex_keys()
        indices = []
        for i in range(len(target_cortex_keys)):
            if structure_tree.structure_descends_from(target_cortex_keys[i],cortex_id):
                indices.append(i)
        
        propagated = []
        for sa in source_areas:
            with open(
                os.path.join(
                    "propagated",
                    f"propagated {sa}",
                ),
                "rb"
            ) as f:
                propagated.append(pickle.load(f))


        test_positions = positions_3d[indices]

        n_voxels_in_columns = []
        dimensions = []

        for pos in tqdm(test_positions):
            ml_coordinates = []
            ap_coordinates = []
            multisensory_weights = []

            streamline = surface_to_surface_streamline(ci,pos)
            dists = shortest_distance(test_positions,streamline)

            for p in propagated:
                ml_c_p = []
                ap_c_p = []
                ms_w_p = []
                for sub_index, dist in zip(indices,dists):
                    dist *= 100
                    if dist <= args.radius_multiplier * avg_std:
                        m = p[sub_index].mean
                        ml_c_p.append(m[0])
                        ap_c_p.append(m[1])
                        ms_w_p.append(p[sub_index].weight)
                ml_coordinates.append(ml_c_p)
                ap_coordinates.append(ap_c_p)
                multisensory_weights.append(ms_w_p)
            
            ml_coordinates = np.array(ml_coordinates)
            ap_coordinates = np.array(ap_coordinates)
            multisensory_weights = np.array(multisensory_weights)
            scaled_ml_coordinates = np.multiply(ml_coordinates, multisensory_weights)
            scaled_ap_coordinates = np.multiply(ap_coordinates, multisensory_weights)
            scaled_coordinates = np.concatenate(
                (scaled_ml_coordinates, scaled_ap_coordinates),
                axis=0
            )

            
            u, s, vh = np.linalg.svd(scaled_coordinates)
            s = s.reshape(1,-1)
            explained_variance_ = (s**2)/((len(source_areas)*2)-1)
            total = np.expand_dims(explained_variance_.sum(axis=1), axis=1)
            
            explained_variance_ratio = explained_variance_ / total
            cumulative_var = np.cumsum(explained_variance_ratio,axis=1)
            
            ev_interp = np.interp(
                0.9,
                np.mean(cumulative_var, axis=0),
                np.arange(1, len(cumulative_var[0]) + 1)
            )

            n_voxels_in_columns.append(ml_coordinates.shape[1])
            dimensions.append(ev_interp)

        radius = round(avg_std*args.radius_multiplier,2)
        plt.figure()
        plt.title(f"Dimensions of {radius}micron-radius columns in {test_area}")
        plt.xlabel("Dimensionality")
        plt.ylabel("# of instances")
        plt.hist(dimensions)
        plt.savefig(f"hist_column_{test_area}.png")

        plt.figure()
        plt.title(f"# of voxels in column vs. Dimensionality: {test_area}")
        plt.xlabel("# of voxels in column")
        plt.ylabel("Dimensionality")
        plt.scatter(n_voxels_in_columns,dimensions)
        plt.savefig(f"scatter_column_{test_area}.png")

if __name__ == "__main__":
    main()