import pickle
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.geodesic_flatmap import GeodesicFlatmap
from deepmouse.maps.map import get_positions
from deepmouse.topography import Gaussian2D

individual_error_bar = False
regenerate_dim_svd = False
cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()

# source_areas = ['SSp-n']
# source_areas = ['SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-n']
source_areas = ["VISp", "AUDp", "SSp-bfd", "SSp-ul", "SSp-m", "SSp-n"]
test_areas = ["FRP"]
test_areas += ["MOp", "MOs"]
test_areas += [
    "SSp-n",
    "SSp-bfd",
    "SSp-ll",
    "SSp-m",
    "SSp-ul",
    "SSp-tr",
    "SSp-un",
    "SSs",
]
test_areas += ["GU", "VISC"]
test_areas += ["AUDd", "AUDp", "AUDpo", "AUDv"]
test_areas += [
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
test_areas += ["ACAd", "ACAv"]
test_areas += ["PL", "ILA"]
test_areas += ["ORBl", "ORBm", "ORBvl"]
test_areas += ["AId", "AIp", "AIv"]
test_areas += ["RSP", "RSPagl", "RSPd", "RSPv"]
test_areas += ["TEa", "PERI"]
test_areas += ["ECT"]


flatmap = GeodesicFlatmap()


def plot_detail(ml_coordinates, ap_coordinates, multisensory_weights, p):
    norm_weights = multisensory_weights / max(multisensory_weights.flatten())

    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=plt.get_cmap("jet"))

    def scatter(coords, weights):
        ax.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            marker=".",
            c=scalar_map.to_rgba(coords),
            alpha=weights,
        )

    for i in range(len(source_areas)):
        ax = fig.add_subplot(2, len(source_areas), i + 1, projection="3d")
        plt.title(source_areas[i])
        scatter(ml_coordinates[i, :], norm_weights[i, :])

        ax = fig.add_subplot(
            2, len(source_areas), i + len(source_areas) + 1, projection="3d"
        )
        scatter(ap_coordinates[i, :], norm_weights[i, :])

    plt.tight_layout()
    plt.show()


data_path = "generated"
if regenerate_dim_svd:
    singular_values = {}
    for test_area in test_areas:
        svalues = []

        positions_3d = get_positions(cache, test_area)
        indices = [flatmap.get_voxel_index(p) for p in positions_3d]
        for omit_experiment_rank in range(4):
            propagated = []
            for sa in source_areas:
                with open(
                    "{}/propagated {} omit {}".format(
                        data_path, sa, omit_experiment_rank
                    ),
                    "rb",
                ) as file:
                    propagated.append(pickle.load(file))

            ml_coordinates = np.zeros((len(source_areas), len(indices)))
            ap_coordinates = np.zeros((len(source_areas), len(indices)))
            multisensory_weights = np.zeros((len(source_areas), len(indices)))
            for i, index in enumerate(indices):
                for j, p in enumerate(propagated):
                    m = p[index].mean
                    ml_coordinates[j, i] = m[0]
                    ap_coordinates[j, i] = m[1]
                    multisensory_weights[j, i] = p[index].weight

            # plot_detail(ml_coordinates, ap_coordinates, multisensory_weights, positions_3d)

            scaled_ml_coordinates = np.multiply(ml_coordinates, multisensory_weights)
            scaled_ap_coordinates = np.multiply(ap_coordinates, multisensory_weights)
            scaled_coordinates = np.concatenate(
                (scaled_ml_coordinates, scaled_ap_coordinates), axis=0
            )

            u, s, vh = np.linalg.svd(scaled_coordinates)
            svalues.append(s)

        explained_variance_ = (np.array(svalues) ** 2) / ((len(source_areas) * 2) - 1)
        total = np.expand_dims(explained_variance_.sum(axis=1), axis=1)
        explained_variance_ratio = explained_variance_ / total
        cumulative_var = np.cumsum(explained_variance_ratio, axis=1)

        cumulative_frac = np.cumsum(np.array(svalues), axis=1).T / np.sum(
            np.array(svalues), axis=1
        )
        singular_values[test_area] = {
            "svalues": svalues,
            "cumulative_frac": cumulative_frac,
            "position": positions_3d,
            "cumulative_var": cumulative_var,
        }
    # singular_values = np.array(singular_values)
    # plt.plot(range(1, len(singular_values)+1), singular_values)
    with open("{}/dimenstion_svd.pkl".format(data_path), "wb") as file:
        pickle.dump(singular_values, file)
else:
    with open("{}/dimenstion_svd.pkl".format(data_path), "rb") as file:
        singular_values = pickle.load(file)


max_sv_area = ""
max_sv = 0
max_ev_area = ""
max_ev = 0
for area in test_areas:
    svalues = np.array(singular_values[area]["svalues"])
    cumulative_var = np.array(singular_values[area]["cumulative_var"])
    cumulative_fraction = np.array(singular_values[area]["cumulative_frac"])
    if individual_error_bar:
        plt.errorbar(
            range(1, len(cumulative_fraction) + 1),
            np.mean(cumulative_fraction, axis=1),
            yerr=np.std(cumulative_fraction, axis=1),
            capsize=3,
        )
        plt.ylabel("Cumulative Fraction")
        # plt.ylim([0, 1])
        plt.legend([area])
        plt.tight_layout()
        plt.savefig("{}/cf_{}.png".format(data_path, area))
        plt.close()

        plt.errorbar(
            range(1, len(svalues[0]) + 1),
            np.mean(svalues, axis=0),
            yerr=np.std(svalues, axis=0),
            capsize=3,
        )
        plt.ylabel("Singular value")
        # plt.ylim([0, 1])
        plt.legend([area])
        plt.tight_layout()
        plt.savefig("{}/sv_{}.png".format(data_path, area))
        plt.close()

        plt.errorbar(
            range(1, len(cumulative_var[0]) + 1),
            np.mean(cumulative_var, axis=0),
            yerr=np.std(cumulative_var, axis=0),
            capsize=3,
        )
        plt.ylabel("Cumulative Explained Variance Ratio")
        # plt.ylim([0, 1])
        plt.legend([area])
        plt.tight_layout()
        plt.savefig("{}/ev_{}.png".format(data_path, area))
        plt.close()

    sv_interp = np.interp(
        0.9,
        np.mean(cumulative_fraction, axis=1),
        np.arange(1, len(cumulative_fraction) + 1),
    )
    if sv_interp > max_sv:
        max_sv = sv_interp
        max_sv_area = area

    ev_interp = np.interp(
        0.9, np.mean(cumulative_var, axis=0), np.arange(1, len(cumulative_var[0]) + 1)
    )
    if ev_interp > max_ev:
        max_ev = ev_interp
        max_ev_area = area

areas_to_plot = ["VISp", "AUDp", "MOp", "SSp-bfd", "SSp-ul", "SSp-m", "RSP"]
# areas_to_plot = ['FRP']
for area in areas_to_plot + [max_sv_area]:
    cumulative_fraction = np.array(singular_values[area]["cumulative_frac"])
    plt.errorbar(
        range(1, len(cumulative_fraction) + 1),
        np.mean(cumulative_fraction, axis=1),
        yerr=np.std(cumulative_fraction, axis=1),
        capsize=3,
    )
plt.ylabel("Cumulative Fraction")
plt.legend(areas_to_plot + [max_sv_area])
plt.tight_layout()
plt.savefig("{}/combine_cf_{}.png".format(data_path, area))
plt.close()

for area in areas_to_plot + [max_ev_area]:
    cumulative_var = np.array(singular_values[area]["cumulative_var"])
    plt.errorbar(
        range(1, len(cumulative_var[0]) + 1),
        np.mean(cumulative_var, axis=0),
        yerr=np.std(cumulative_var, axis=0),
        capsize=3,
    )

plt.ylabel("Cumulative Explained Variance Ratio")
plt.legend(areas_to_plot + [max_ev_area])
plt.tight_layout()
plt.savefig("{}/combine_ev_{}.png".format(data_path, area))
plt.close()


position_xyz = []
sv_values_90 = []
ev_values_90 = []
positions = []
sv_vals = []
ev_vals = []
msize = []
txt_labels = {}
for area in test_areas:
    cumulative_fraction = np.array(singular_values[area]["cumulative_frac"])
    positions_3d = np.array(
        singular_values[area]["position"]
    )  # get_positions(cache, area)
    cumulative_var = np.array(singular_values[area]["cumulative_var"])
    ev_interp = np.interp(
        0.9, np.mean(cumulative_var, axis=0), np.arange(1, len(cumulative_var[0]) + 1)
    )

    sv_interp = np.interp(
        0.9,
        np.mean(cumulative_fraction, axis=1),
        np.arange(1, len(cumulative_fraction) + 1),
    )
    sv_vals.extend([sv_interp] * len(positions_3d))
    ev_vals.extend([ev_interp] * len(positions_3d))
    msize.extend([0.01] * len(positions_3d))
    positions.extend(list(positions_3d))
    position_xyz.append(np.mean(positions_3d, axis=0))
    sv_values_90.append(sv_interp)
    ev_values_90.append(ev_interp)
    if area in source_areas:
        txt_labels[area] = np.mean(positions_3d, axis=0)
positions = np.array(positions)
sv_vals = np.array(sv_vals)
ev_vals = np.array(ev_vals)
msize = np.array(msize)
position_xyz = np.array(position_xyz)
sv_values_90 = np.array(sv_values_90)
# Creating figure

comb_position_xyz = np.concatenate([positions, position_xyz], axis=0)
comb_sv_values_90 = np.concatenate([sv_vals, sv_values_90], axis=0)
comb_ev_values_90 = np.concatenate([ev_vals, ev_values_90], axis=0)
comb_msize = np.concatenate([msize, [10] * len(sv_values_90)], axis=0)

positions_list = [positions, positions, comb_position_xyz, comb_position_xyz]
values = [sv_vals, ev_vals, comb_sv_values_90, comb_ev_values_90]
labels = [
    "cumulative fraction",
    "explained variance",
    "cumulative fraction",
    "explained variance",
]
dim_sizes = [msize, msize, comb_msize, comb_msize]
for idx in range(len(values)):
    position_3d = positions_list[idx]
    vals = values[idx]
    label = labels[idx]
    msize = dim_sizes[idx]
    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")

    # Add x, y gridlines
    ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)

    # Creating color map
    my_cmap = plt.get_cmap("plasma")

    # Creating plot
    sctt = ax.scatter3D(
        position_3d[:, 0],
        position_3d[:, 2],
        position_3d[:, 1],
        alpha=1,
        c=vals,
        cmap=my_cmap,
        s=msize,
        marker="o",
    )

    for area, pos in txt_labels.items():
        ax.text(pos[0], pos[2], pos[1], f"{area}", fontsize=7)

    from textwrap import wrap

    title = ax.set_title(f"Dimension based on 90% {label}")
    ax.axes.yaxis.set_ticklabels([])
    ax.invert_zaxis()
    ax.set_yticklabels([])
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(azim=-83, elev=1)
    plt.tight_layout()
    plt.savefig("{}/dimension_{}_{}.png".format(data_path, str(idx), label))
    plt.close()
