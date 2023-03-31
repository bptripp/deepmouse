import pickle
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from deepmouse.maps.util import get_voxel_model_cache, get_default_structure_tree
from deepmouse.dimension import get_test_areas

data_path = "propagated"
data_path_mixed = "propagated_mixing"
data_path_layer = "propagated_layer"
data_path_omit = "propagated"
cache = get_voxel_model_cache()
structure_tree = get_default_structure_tree()

source_areas = ["VISp", "AUDp", "SSp-bfd", "SSp-ul", "SSp-m", "SSp-n"]
test_areas = get_test_areas()


def compute_dim_pr(area, singular_values1, singular_values2, plot_type, granularity="area"):
    layer_names = []
    ev_interp2 = None
    if granularity == "area":
        if plot_type == "Dimension":
            cumulative_var = np.array(singular_values1[area]["cumulative_var"])
            ev_interp1 = np.interp(
                0.9, np.mean(cumulative_var, axis=0), np.arange(
                    1, len(cumulative_var[0]) + 1)
            )
        else:
            sval1 = np.array(singular_values1[area]["svalues"])
            if len(sval1) == 1:
                ev_interp1 = np.sum(sval1)**2/np.sum(sval1**2)
            else:
                ev_interp1 = np.mean(np.sum(sval1, axis=1)
                                     ** 2/(np.sum(sval1**2, axis=1)))

        if singular_values2 is not None:
            if plot_type == "Dimension":
                cumulative_var2 = np.array(
                    singular_values2[area]["cumulative_var"])
                ev_interp2 = np.interp(0.9, np.mean(cumulative_var2, axis=0),
                                       np.arange(1, len(cumulative_var2[0]) + 1))
            else:
                sval2 = np.array(singular_values2[area]["svalues"])
                if len(sval2) == 1:
                    ev_interp2 = np.sum(sval2)**2/np.sum(sval2**2)
                else:
                    ev_interp2 = np.mean(
                        np.sum(sval2, axis=1)**2/(np.sum(sval2**2, axis=1)))

    else:
        ev_interp1 = []

        for layer in ["1", "2/3", "4", "5", "6a", "6b"]:
            area_layer = area+layer
            if area_layer in structure_tree.get_id_acronym_map():
                cumulative_var = np.array(
                    singular_values1[area_layer]["cumulative_var"])
                if len(cumulative_var[0]) < 1:
                    continue
                if plot_type == "Dimension":
                    ev_interp1 = np.interp(0.9, np.mean(
                        cumulative_var, axis=0), np.arange(1, len(cumulative_var[0]) + 1))
                else:
                    svalues = np.array(
                        singular_values1[area_layer]["svalues"])[0, :]
                    ev_interp = np.sum(svalues)**2/(np.sum(svalues**2))

                ev_interp1.append(ev_interp)
                layer_names.append(layer)

    return ev_interp1, ev_interp2, layer_names


def generate_dim_pr_plot(singular_values1, singular_values2, labels, plot_type, granularity, colors=[], fname="", data_path=None):
    ev_vals = []
    return_colors = True if not colors else False

    plt.figure(figsize=(12, 6))
    point_loc, point_loc_prev = 0, 0
    areas = []
    ev_dict1, ev_dict2 = {}, {}
    for idx, area in enumerate(test_areas):
        ev_interp1, ev_interp2, layer_names = compute_dim_pr(
            area, singular_values1, singular_values2, plot_type, granularity)

        areas.append(area)
        if granularity == "area":
            point_loc += 1
            ev_vals.append(ev_interp1)
            ev_dict1[area] = ev_interp1
        else:
            ev_vals.extend(ev_interp1)
            point_loc += len(ev_interp1)
            areas.extend([""]*(len(ev_interp1)-1))
            ev_dict1[area] = ev_interp1
            ev_dict1[f"{area}_layer"] = layer_names

        markerline, stemlines, baseline = plt.stem(np.arange(
            point_loc_prev, point_loc), ev_interp1, markerfmt='o', label=labels[0])
        if return_colors:
            color = plt.getp(markerline, 'color')
            colors.append(color)
        else:
            color = colors[idx]
            plt.setp(markerline, 'color', color)

        plt.setp(stemlines, 'color', color)
        plt.setp(stemlines, 'linestyle', 'solid')
        if singular_values2 is not None:
            plt.plot(np.arange(point_loc, point_loc+1),
                     ev_interp2, 'x', label=labels[1])
            ev_dict2[area] = ev_interp2
        point_loc_prev = point_loc

    plt.xticks(np.arange(1, len(ev_vals)+1), areas, rotation=90)
    plt.ylabel(f"{plot_type}")
    plt.title(
        f"Comparison of {plot_type} of different {granularity} with propagation and/or mixing")
    plt.savefig(
        f"{data_path}/{plot_type}_{granularity}_{fname}_compare.png", transparent=True)

    plt.close()
    return colors, ev_dict1, ev_dict2


def get_pr_layers(ev_interp, layers, layers_dict):
    ev_vals = []
    for layer in layers:
        if layer in layers_dict:
            idx = layers_dict.index(layer)
            ev_vals.append(ev_interp[idx])
    return ev_vals


def plot_hierarchy_correlate(ev_interp_val, hscores_cc_ct_sort, data_path, area_layer="area", layers_dict={}, fname=""):

    ev_vals_cc_ct = []
    hierarchy_score_CC_TC_CT = []
    cc_tc_ct_areas = []
    CC_TC_CT_scores = hscores_cc_ct_sort.loc[:, ["Cre_conf CC+TC+CT"]].values
    for idx, area in enumerate(hscores_cc_ct_sort.loc[:, ["Area"]].values):
        area = area[0]
        if area_layer == "area":
            ev_vals_cc_ct.append(ev_interp_val[area])
        else:
            edge_vals = get_pr_layers(ev_interp_val[area], [
                                      "1", "6a", "6b"], ev_interp_val[f"{area}_layer"])
            mid_vals = get_pr_layers(ev_interp_val[area], [
                                     "2/3", "4", "5"], ev_interp_val[f"{area}_layer"])
            ev_vals_cc_ct.append(np.mean(edge_vals) - np.mean(mid_vals))

        hierarchy_score_CC_TC_CT.append(CC_TC_CT_scores[idx][0])
        cc_tc_ct_areas.append(area)

    hierarchy_score_CC_TC_CT = np.array(hierarchy_score_CC_TC_CT)
    ev_vals_cc_ct = np.array(ev_vals_cc_ct)

    plt.figure()
    plt.scatter(hierarchy_score_CC_TC_CT, ev_vals_cc_ct, linewidths=5)
    x, y = hierarchy_score_CC_TC_CT, ev_vals_cc_ct

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    t = stats.t.ppf(0.975, x.size-2)

    plt.plot(x, slope*x+intercept, linewidth=4)
    plt.xticks(np.arange(-0.4, 0.6, 0.3))
    plt.yticks(np.arange(0.0, 6, 2.5))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("Hierarchy Score", fontsize=25)
    plt.ylabel("Participation Ratio", fontsize=25)
    # plt.title("Hierarchy vs Dimension based on Propagation and/or Mixing",fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"{data_path}/hierarchy_{fname}_{slope-t*std_err}_{slope+t*std_err}.png", transparent=True)
    plt.close()


if __name__ == "__main__":

    with open("{}/dimenstion_svd_single.pkl".format(data_path), "rb") as file:
        singular_values = pickle.load(file)

    with open("{}/dimenstion_svd_single.pkl".format(data_path_mixed), "rb") as file:
        singular_values_mixed = pickle.load(file)

    with open("{}/dimenstion_svd_single.pkl".format(data_path_layer), "rb") as file:
        singular_values_layer = pickle.load(file)

    with open("{}/dimenstion_svd_omit.pkl".format(data_path_omit), "rb") as file:
        singular_values_omit = pickle.load(file)

    for test_area in test_areas:
        singular_values_omit[test_area]["svalues"].extend(
            singular_values[test_area]["svalues"])
        singular_values_omit[test_area]["cumulative_frac"] = np.concatenate(
            [singular_values_omit[test_area]["cumulative_frac"], singular_values[test_area]["cumulative_frac"]], axis=1)
        singular_values_omit[test_area]["cumulative_var"] = np.concatenate(
            [singular_values_omit[test_area]["cumulative_var"], singular_values[test_area]["cumulative_var"]], axis=0)

    hscore = pd.read_csv("generated/hierarchy.csv")
    hscores_cc_ct_sort = hscore.sort_values(by=['Cre_conf CC+TC+CT'])

    labels = ["Propagated", "Propagated and Mixed"]
    colors, _, _ = generate_dim_pr_plot(singular_values, singular_values_mixed,
                                        labels, "Dimension", "area", colors=[], fname="single", data_path=data_path)
    _, pr_val, pr_val_mixed = generate_dim_pr_plot(
        singular_values, singular_values_mixed, labels, "PR", "area", colors=colors, fname="single", data_path=data_path)
    _, pr_val_omit, _ = generate_dim_pr_plot(
        singular_values_omit, None, labels[0], "PR", "area", colors=colors, fname="omit", data_path=data_path_omit)
    _, pr_val_layer, _ = generate_dim_pr_plot(
        singular_values_layer, None, labels[0], "PR", "layer", colors=colors, fname="single", data_path=data_path_layer)

    plot_hierarchy_correlate(pr_val, hscores_cc_ct_sort,data_path, fname="single")
    plot_hierarchy_correlate(pr_val_mixed, hscores_cc_ct_sort, data_path_mixed, fname="single")
    plot_hierarchy_correlate(pr_val_omit, hscores_cc_ct_sort, data_path_omit, fname="omit")
    plot_hierarchy_correlate(pr_val_layer, hscores_cc_ct_sort,data_path_layer, area_layer="layer", fname="single")
