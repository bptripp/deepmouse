import numpy as np
import os
import pickle
from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser
from find_voxel_cortical_area import get_target_cortex_keys, get_voxel_same_area_indices
from horizontal_distance import shortest_distance, surface_to_surface_streamline
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.maps.map import right_target_indices, get_positions
from streamlines import CompositeInterpolator
from topography import get_primary_gaussians, propagate_gaussians_through_isocortex, GaussianMixture2D, Gaussian2D


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--omit_experiment_rank", "-o", default=None)
    # Directory containing weights and nodes
    parser.add_argument("--data_dir", "-d", default="data_files")

    # If propagated Gaussians files exist, pass in that directory
    # Directory containing ONLY pickle files of propagated Gaussians
    parser.add_argument("--prop_dir", "-p", default=None)

    # If propagated Gaussians must be generated first, pass in a list of cortical areas
    # List of source cortical areas (e.g. -a VISp AUDp SSp-n)
    parser.add_argument("--areas_list", "-a", default=None, nargs="+")
    args = parser.parse_args()
    # Assert that only one of prop_dir and areas_list has been passed
    assert bool(args.prop_dir) != bool(args.areas_list)

    return args

# Assign a Gaussian-based weighting based on lateral distance to streamline


def gaussian_weighting(dist, std_dev):
    A = 1 / (std_dev * np.sqrt(2 * np.pi))
    base = np.exp(-0.5 * np.square(dist / std_dev))

    return A * base


def mix_in_column(
    propagated,       # List of Gaussians for target voxels in the right isocortex
    # List of positions that correspond to each propagated Gaussian in "propagated"
    target_positions,
    ci,               # CompositeInterpolator object
    # Number of standard deviations of distance that is considered in the weighting
    min_std_dist=3,
):

    # Matrix of standard deviations of the interlaminar connections from the CNN Mousenet paper
    il_stds = np.array([
        [142.5,    31.67,   63.33],
        [139.33,   114,     88.67],
        [95,       63.33,   133]
    ])

    # Find the average standard deviation to use as the std_dev parameter of the Gaussian weighting scheme
    avg_std = np.mean(il_stds)

    results_mixed = []  # List of propagated and mixed voxels for this area

    # Find the parent ID key of each target cortex
    target_cortex_keys = get_target_cortex_keys()  # (61878, )

    propagated = np.array(propagated)  # (61878,)

    for voxel in tqdm(target_positions):

        # Streamline of the voxel
        streamline = surface_to_surface_streamline(ci, voxel)
        voxel_mixture = GaussianMixture2D()  # Instantiate Mixture object

        # Get the target voxels within the same cortical area
        area_indices = get_voxel_same_area_indices(
            voxel, target_positions, target_cortex_keys).astype(np.int32)

        assert area_indices[-1] < len(target_positions)
        target_positions_sa = target_positions[area_indices]
        propagated_sa = propagated[area_indices]
        assert len(target_positions_sa) == len(propagated_sa)

        # Find shortest distance between each propagated voxel and the streamline
        dists = shortest_distance(target_positions_sa, streamline)

        indices = np.where(dists*100 <= min_std_dist * avg_std)[0]

        # Iterate through all of the other propagated voxels of this area
        for index in indices:
            gaussian, dist = propagated_sa[index], dists[index]

            dist *= 100  # Convert dist from voxel units to microns

            # If the distant voxel is within 3 standard devs (~290 microns) of the streamline, add it to the mixture
            # if dist <= min_std_dist * avg_std:
            weight = gaussian_weighting(dist, avg_std)
            gauss_weight = gaussian.weight*weight
            mean = deepcopy(gaussian.mean)
            covariance = deepcopy(gaussian.covariance)
            voxel_mixture.add(Gaussian2D(gauss_weight, mean, covariance))

        # Find mixture approximation for the voxel and add it to the results list for this area
        results_mixed.append(voxel_mixture.approx())

    return results_mixed


def main():
    # Parse command-line arguments
    args = parse_args()

    # Instantiate voxel model cache and structure tree
    cache = get_voxel_model_cache()
    structure_tree = get_default_structure_tree()

    with open('interpolator.pkl', 'rb') as file:
        ci = pickle.load(file)

    # If a directory with propagated Gaussians in pickle files already exists, just load them and mix
    if args.prop_dir:
        print(f"Loading propagated files from directory '{args.prop_dir}'")
        # Find the target indices of the right isocortex
        cortex_id = structure_tree.get_id_acronym_map()['Isocortex']
        target_mask = cache.get_target_mask()
        target_keys = target_mask.get_key(structure_ids=None)  # (448962,)
        target_cortex_indices = []
        for i in range(len(target_keys)):
            if structure_tree.structure_descends_from(target_keys[i], cortex_id):
                target_cortex_indices.append(i)

        # Find the target positions of the right isocortex
        target_cortex_indices = np.array(target_cortex_indices)  # (123245,)
        r = right_target_indices(cache)  # (61878, )
        right_target_cortex_indices = target_cortex_indices[r]
        target_positions = get_positions(cache, "root", True)[
            right_target_cortex_indices]  # (61878, 3)
        os.makedirs(f"{args.prop_dir}_mixing", exist_ok=True)
        # Run mixing for each file of propagated Gaussians in the directory
        for prop_fname in os.listdir(args.prop_dir):
            if "propagated" not in prop_fname:
                continue
            print(f"Loading {prop_fname}")
            pickle_file = open(os.path.join(args.prop_dir, prop_fname), "rb")
            propagated = pickle.load(pickle_file)
            pickle_file.close()

            propagated_mixed = mix_in_column(
                propagated,
                target_positions,
                ci,
            )

            # Dump propagated + mixed voxels for this area into a pickle file
            with open(
                f"{args.prop_dir}_mixing/propagated_and_mixed_{prop_fname}.pkl",
                "wb"
            ) as file:
                pickle.dump(propagated_mixed, file)
                file.close()

            with open(
                "JUST_IN_CASE.pkl",
                "wb"
            ) as file_backup:
                pickle.dump(propagated_mixed, file_backup)
                file_backup.close()

    # Otherwise, accept a list of areas, propagate them through the isocortex first and then mix them
    else:
        if not os.path.exists("propagated"):
            os.mkdir("propagated")

        for area in args.areas_list:
            # Get gaussians and positions of the voxels in the area
            gaussians, positions_3d = get_primary_gaussians(area)
            propagated, target_positions = propagate_gaussians_through_isocortex(
                gaussians, positions_3d, args.data_dir)

            propagated_mixed = mix_in_column(
                propagated,
                target_positions,
                ci,
            )

            # Dump propagated + mixed vowels for this area into a pickle file
            with open(
                # os.path.join("propagated",f"propagated_and_mixed_{area}.pkl"),
                f"propagated_and_mixed_{area}.pkl",
                "wb"
            ) as file:
                pickle.dump(propagated_mixed, file)


if __name__ == "__main__":
    main()
