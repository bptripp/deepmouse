import numpy as np
import os
import pickle
from tqdm import tqdm

from argparse import ArgumentParser
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.maps.map import right_target_indices, get_positions
from geodesic_flatmap import GeodesicFlatmap
from find_voxel_cortical_area import get_target_positions, get_voxel_same_area_indices
from horizontal_distance import shortest_distance, surface_to_surface_streamline
import matplotlib.pyplot as plt
from deepmouse.dimension import get_test_areas


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--radius_multiplier", "-r", default=3, type=int)

    parser.add_argument("--type", "-t", default=0, type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_path = ["propagated", "propagated_mixing"][args.type]
    prefix = ["", "propagated_and_mixed_"][args.type]
    extension = ["", ".pkl"][args.type]
    plots_dir = ["plots", "plots_mixing"][args.type]
    source_areas = ["VISp", "AUDp", "SSp-bfd", "SSp-ul", "SSp-m", "SSp-n"]
    test_areas = get_test_areas()
    il_stds = np.array([[142.5, 31.67, 63.33], [139.33, 114, 88.67], [95, 63.33, 133]])

    avg_std = np.mean(il_stds)

    cache = get_voxel_model_cache()
    structure_tree = get_default_structure_tree()

    # CORRECT METHOD
    target_positions = get_target_positions()
    with open("interpolator.pkl", "rb") as file:
        ci = pickle.load(file)

    propagated = []
    for sa in source_areas:
        with open(
            f"{data_path}/{prefix}propagated {sa} omit {0}{extension}",
            "rb",
        ) as file:
            propagated.append(pickle.load(file))
    os.makedirs(plots_dir, exist_ok=True)
    flatmap = GeodesicFlatmap()
    # curr_plots = os.listdir(args.plots_dir)

    for test_area in tqdm(test_areas):
        print(f"Analyzing {test_area}...")
        test_positions = get_positions(cache, test_area)
        indices = [flatmap.get_voxel_index(p) for p in test_positions]
        n_voxels_in_columns = []
        dimensions = []
        voxel_positions = target_positions[indices]
        for pos in tqdm(voxel_positions):
            ml_coordinates = []
            ap_coordinates = []
            multisensory_weights = []

            streamline = surface_to_surface_streamline(ci, pos)
            dists = shortest_distance(voxel_positions, streamline)

            for p in propagated:
                ml_c_p = []
                ap_c_p = []
                ms_w_p = []
                column_indices = np.where(dists * 100 <= args.radius_multiplier * avg_std)[0]

                for sub_index in column_indices:
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
                (scaled_ml_coordinates, scaled_ap_coordinates), axis=0
            )

            u, sv, vh = np.linalg.svd(scaled_coordinates)
            sv = sv.reshape(1, -1)
            pr_interp = np.sum(sv) ** 2 / (np.sum(sv**2))

            n_voxels_in_columns.append(ml_coordinates.shape[1])
            dimensions.append(pr_interp)

        radius = round(avg_std * args.radius_multiplier, 2)

        plt.figure()
        plt.title(f"PR of {radius}micron-radius columns in {test_area}")
        plt.xlabel("Participation Ratio")
        plt.ylabel("# of instances")
        plt.hist(dimensions)
        plt.savefig(f"{plots_dir}/hist_column_{test_area}.png")
        plt.close()
        plt.figure()
        plt.title(f"# of Voxels in Column vs. PR: {test_area}")
        plt.xlabel("# of Voxels in Column")
        plt.ylabel("Participation Ratio")
        plt.scatter(n_voxels_in_columns, dimensions)
        plt.savefig(f"{plots_dir}/scatter_column_{test_area}.png")
        plt.close()


if __name__ == "__main__":

    main()
