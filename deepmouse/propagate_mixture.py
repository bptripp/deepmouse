import numpy as np
import os
import pickle
from tqdm import tqdm

from argparse import ArgumentParser
from horizontal_distance import shortest_distance, surface_to_surface_streamline
from maps.util import get_voxel_model_cache, get_default_structure_tree
from maps.map import right_target_indices, get_positions
from streamlines import CompositeInterpolator
from topography import get_primary_gaussians, propagate_gaussians_through_isocortex, GaussianMixture2D, Gaussian2D

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--omit_experiment_rank","-o",default=None)
    parser.add_argument("--data_dir","-d",default="data_files") # Directory containing weights and nodes

    # If propagated Gaussians files exist, pass in that directory
    parser.add_argument("--prop_dir","-p",default=None) # Directory containing ONLY pickle files of propagated Gaussians

    # If propagated Gaussians must be generated first, pass in a list of cortical areas
    parser.add_argument("--areas_list","-a",default=None, nargs="+") # List of source cortical areas
    args = parser.parse_args()
    assert bool(args.prop_dir) != bool(args.areas_list) # Assert that only one of prop_dir and areas_list has been passed

    return args

def gaussian_weighting(
    dist,
    std_dev
):
    A = 1 / (std_dev * np.sqrt(2 * np.pi))
    base = np.exp(
        -0.5 * np.square(dist / std_dev)
    )

    return A * base

def mix_in_column(
    propagated, # List of Gaussians for target voxels in the right isocortex
    target_positions, # List of positions that correspond to each propagated Gaussian in "propagated"
    ci, # CompositeInterpolator object
    min_std_dist=3, # Number of standard deviations of distance that is considered in the weighting
):

    # Matrix of standard deviations of the interlaminar connections from the CNN Mousenet paper
    il_stds = np.array([
        [142.5,    31.67,   63.33],
        [139.33,   114,     88.67],
        [95,       63.33,   133]
    ])

    # Find the average standard deviation to use as the std_dev parameter of the Gaussian weighting scheme
    avg_std = np.mean(il_stds)

    results_mixed = [] # List of propagated and mixed voxels for this area

    for voxel in tqdm(target_positions):

        streamline = surface_to_surface_streamline(ci,voxel) # Streamline for voxel
        voxel_mixture = GaussianMixture2D() # Instantiate Mixture object
        
        # Find shortest distance between each propagated voxel and the streamline
        dists = shortest_distance(target_positions,streamline)

        # Iterate through all of the other propagated voxels of this area
        for gaussian, dist in zip(propagated, dists):
            
            dist *= 100 # Convert dist from voxel units to microns

            # If the distant voxel is within 3 standard devs (~290 microns) of the streamline, add it to the mixture
            if dist <= min_std_dist * avg_std:
                weight = gaussian_weighting(dist, avg_std)
                gaussian.weight = weight
                voxel_mixture.add(gaussian)

        # Find mixture approximation for the voxel and add it to the results list for this area
        results_mixed.append(voxel_mixture.approx())

    return results_mixed

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Instantiate voxel model cache and structure tree
    cache = get_voxel_model_cache()
    structure_tree = get_default_structure_tree()

    from streamlines import CompositeInterpolator
    with open('interpolator.pkl', 'rb') as file:
        ci = pickle.load(file)

    # If a directory with propagated Gaussians in pickle files already exists, just load them and mix
    if args.prop_dir:

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

        for prop_fname in os.listdir(args.prop_dir):
            pickle_file = open(os.path.join(args.prop_dir,prop_fname),"rb")
            propagated = pickle.load(pickle_file)
            pickle_file.close()

            propagated_mixed = mix_in_column(
                propagated,
                target_positions,
                ci,
            )

            # Dump propagated + mixed voxels for this area into a pickle file
            with open(
                os.path.join(args.prop_dir,f"{prop_fname}_mixed.pkl"),
                "wb"
            ) as file:
                pickle.dump(propagated_mixed, file)

    # Otherwise, accept a list of areas, propagate them through the isocortex first and then mix them
    else:
        if not os.path.exists("propagated"):
            os.mkdir("propagated")

        for area in args.areas_list:
            # Get gaussians and positions of the voxels in the area
            gaussians, positions_3d = get_primary_gaussians(area)
            propagated, target_positions = propagate_gaussians_through_isocortex(gaussians,positions_3d,args.data_dir)

            propagated_mixed = mix_in_column(
                propagated,
                target_positions,
                ci,
            )

            # Dump propagated + mixed vowels for this area into a pickle file
            with open(
                os.path.join("propagated",f"propagated_and_mixed_{area}.pkl"),
                "wb"
            ) as file:
                pickle.dump(propagated_mixed, file)

if __name__ == "__main__":
    main()