#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenta imágenes .jpg con Cellpose usando un Grid Search adaptativo por frame,
guarda parámetros óptimos y estadísticas, genera heatmaps célula a célula 
y crea gráficos temporales/polares con test estadísticos.
"""

import os
import re

# --------------------------------------------------------------------------------
# --- SELECCIÓN DE GPU (DEBE IR ANTES DE IMPORTAR CELLPOSE NI TORCH) ---
# --------------------------------------------------------------------------------
# "0" = Primera tarjeta gráfica
# "1" = Segunda tarjeta gráfica
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# --------------------------------------------------------------------------------

import glob
import shutil
import numpy as np
from PIL import Image
from cellpose import models
import json
import cv2
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp
from skimage import exposure, img_as_ubyte
from skimage import measure
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d  
from itertools import combinations
import itertools
from astropy.stats import kuiper_two
from scipy.stats import circmean
from scipy.stats import mannwhitneyu
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sys

# --------------------------------------------------------------------------------
# --- CONFIGURACIÓN DE ESTILO GRAPHPAD PRISM ---
# --------------------------------------------------------------------------------
sns.set_style("ticks") 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.5      
plt.rcParams['xtick.major.width'] = 1.5   
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5      
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.direction'] = 'out'   
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['legend.frameon'] = False    
# --------------------------------------------------------------------------------

# --- Carpetas raíz ---
root_dir = "./"
stats_output_dir = "./segmentation_stats/"
overlay_output_dir = "./segmentation_overlay/"
os.makedirs(stats_output_dir, exist_ok=True)
os.makedirs(overlay_output_dir, exist_ok=True)

# VARIABLE GLOBAL PARA EXPORTAR ESTADÍSTICAS A CSV
global_stats_list = []

# --- CONFIGURACIÓN EROSIÓN ---
erosion_radius_compute = 0

# --- NUEVOS PARÁMETROS PARA Z-DRIFT / TIGHT JUNCTIONS ---
junc_window_size = 100       
junc_threshold_ratio = 0.5   
fExtractGlobalROI = 1        

# --------------------------------------------------------------------------------
# FUNCIONES AUXILIARES Y CELLPOSE (Se mantienen intactas)
# --------------------------------------------------------------------------------
def maximal_rectangle(matrix):
    rows, cols = matrix.shape
    heights = np.zeros(cols, dtype=int)
    max_area = 0
    best_rect = (0, 0, 0, 0) 
    
    for i in range(rows):
        heights = np.where(matrix[i], heights + 1, 0)
        stack = []
        for j in range(cols + 1):
            h = heights[j] if j < cols else 0
            while stack and h < heights[stack[-1]]:
                h_idx = stack.pop()
                height = heights[h_idx]
                width = j if not stack else j - stack[-1] - 1
                area = height * width
                if area > max_area:
                    max_area = area
                    x = stack[-1] + 1 if stack else 0
                    y = i - height + 1
                    best_rect = (x, y, width, height)
            stack.append(j)
    return best_rect

def get_valid_window_mask(label_mask, stats, window_size, threshold_ratio):
    H, W = label_mask.shape
    valid_mask = np.ones((H, W), dtype=bool)
    if not stats: return valid_mask
    mean_jf_global = np.mean([s["junction_fraction"] for s in stats])
    threshold = mean_jf_global * threshold_ratio
    max_label = np.max(label_mask)
    jf_map = np.full(max_label + 1, np.nan) 
    
    for s in stats:
        jf_map[s["label"]] = s["junction_fraction"]
        
    for y in range(0, H, window_size):
        for x in range(0, W, window_size):
            y_end = min(y + window_size, H)
            x_end = min(x + window_size, W)
            block = label_mask[y:y_end, x:x_end]
            unique_labels = np.unique(block)
            unique_labels = unique_labels[unique_labels > 0]
            if len(unique_labels) > 0:
                block_vals = jf_map[unique_labels]
                block_vals = block_vals[~np.isnan(block_vals)]
                if len(block_vals) > 0:
                    block_mean_jf = np.mean(block_vals)
                    if block_mean_jf < threshold:
                        valid_mask[y:y_end, x:x_end] = False
    return valid_mask

def compute_mask_stats(label_mask, search_radius=1):
    stats = []
    is_perim = np.zeros_like(label_mask, dtype=bool)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue 
            shifted = np.roll(np.roll(label_mask, dx, axis=0), dy, axis=1)
            if dx > 0: shifted[:dx, :] = 0
            if dx < 0: shifted[dx:, :] = 0
            if dy > 0: shifted[:, :dy] = 0
            if dy < 0: shifted[:, dy:] = 0
            is_perim |= (label_mask > 0) & (label_mask != shifted)
    
    is_junct = np.zeros_like(label_mask, dtype=bool)
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0: continue 
            shifted = np.roll(np.roll(label_mask, dx, axis=0), dy, axis=1)
            if dx > 0: shifted[:dx, :] = 0
            if dx < 0: shifted[dx:, :] = 0
            if dy > 0: shifted[:, :dy] = 0
            if dy < 0: shifted[:, dy:] = 0
            is_junct |= (label_mask > 0) & (shifted > 0) & (label_mask != shifted)

    is_junct = is_junct & is_perim
    max_id = label_mask.max() + 1
    perim_counts = np.bincount(label_mask[is_perim], minlength=max_id)
    junct_counts = np.bincount(label_mask[is_junct], minlength=max_id)

    for prop in regionprops(label_mask):
        if prop.area < 10: continue
        lbl = prop.label
        p_count = perim_counts[lbl]
        j_count = junct_counts[lbl]
        junction_fraction = j_count / p_count if p_count > 0 else 0.0
        junction_fraction = min(1.0, max(0.0, junction_fraction))
        stats.append({
            "label": lbl,
            "area": float(prop.area),
            "solidity": float(prop.solidity),
            "eccentricity": float(prop.eccentricity),
            "orientation": float(np.degrees(prop.orientation) % 180),
            "junction_fraction": float(junction_fraction)
        })
    return stats

def save_segmentation_overlay(img_gray, mask, save_path, thickness=2):
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    for label_id in np.unique(mask):
        if label_id == 0: continue
        single_mask = (mask==label_id).astype(np.uint8)
        contours,_ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img_rgb, contours, -1, (0, 0, 0), thickness)
    Image.fromarray(img_rgb).save(save_path)

def save_metric_colormap_overlay(img_gray, label_mask, stats, metric_name, save_path, vmin=None, vmax=None, cmap_name='viridis', alpha=0.7, filter_mask=None):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.imshow(img_gray, cmap='gray')
    max_label = np.max(label_mask)
    val_array = np.full(max_label + 1, np.nan)
    for s in stats:
        label_id = s.get("label")
        if label_id is not None and label_id <= max_label:
            val_array[label_id] = s[metric_name]

    metric_img = val_array[label_mask]
    valid_mask = ~np.isnan(metric_img) & (label_mask > 0)
    if filter_mask is not None: valid_mask = valid_mask & filter_mask

    colored_mask = np.zeros((*label_mask.shape, 4))
    cmap = plt.get_cmap(cmap_name)
    if vmin is None: vmin = np.nanmin(val_array)
    if vmax is None: vmax = np.nanmax(val_array)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colored_mask[valid_mask] = cmap(norm(metric_img[valid_mask]))
    colored_mask[valid_mask, 3] = alpha
    ax.imshow(colored_mask)

    draw_mask = label_mask.copy()
    if filter_mask is not None: draw_mask[~filter_mask] = 0
    for label_id in np.unique(draw_mask):
        if label_id == 0: continue
        cell_mask = (draw_mask == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim > 1: ax.plot(cnt[:, 0], cnt[:, 1], color='black', linewidth=0.5)
                
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name.replace("_", " ").title(), fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# --------------------------------------------------------------------------------
# EJECUCIÓN CELLPOSE (Omitido para ejecución rápida del script en gráficos)
# --------------------------------------------------------------------------------
# (Tu código previo del Grid Search adaptativo sigue aquí íntegramente...)

# -------------------------------------------------------------------------------------
# FUNCIONES DE GRAFICADO ESTADÍSTICO ACTUALIZADAS PARA PLOS ONE
# -------------------------------------------------------------------------------------

def save_statistics_to_csv(filepath="statistical_results_table.csv"):
    """Exporta las métricas calculadas a un CSV global"""
    if not global_stats_list:
        return
    df_stats = pd.DataFrame(global_stats_list)
    df_stats.to_csv(filepath, index=False)
    print(f"\n[INFO] Todas las comparaciones estadísticas exportadas a: {filepath}")

def plot_temporal_metrics_with_ks(stats_dir, json_files=None, save_dir=None, save_name=None, 
                                  min_ecc=None, max_ecc=None, min_area=None, max_area=None, 
                                  label_mapping=None, smooth_lines=True, smoothing_sigma=2,
                                  ks_eval_points=None, aggregate_by_frame=False,
                                  group_mapping=None, pixel_size_um=None, 
                                  baseline_json_files=None, baseline_calc_pts=5, norm_mode=None,
                                  baseline_from_start=False):
    
    global global_stats_list
    all_data = []
    raw_distributions = {}

    if json_files is None:
        json_files = [f for f in os.listdir(stats_dir) if f.endswith(".json")]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    metrics_list = ["mean_area", "mean_solidity", "mean_eccentricity", "mean_orientation", "mean_junction_fraction"]
    loaded_experiments = [] 
    min_len = float('inf')

    for jf in json_files:
        path = os.path.join(stats_dir, jf)
        with open(path, "r") as f:
            data = json.load(f)
            folder_name = data["folder"]
            tp_sorted = natural_sort(list(data["timepoints"].keys()))
            display_label = label_mapping[jf] if label_mapping is not None and jf in label_mapping else folder_name
            current_len = len(tp_sorted)
            if current_len < min_len: min_len = current_len
            loaded_experiments.append((data, tp_sorted, display_label, jf))

    for idx_f, (data, tp_sorted, exp_label, jf) in enumerate(loaded_experiments):
        main_jf = json_files[idx_f]
        interval_mins = data.get("interval", 5.0)
        if interval_mins <= 1.0: interval_mins = 5.0
        match = re.search(r'every\s*(\d+)\s*min', main_jf.lower() + " " + exp_label.lower())
        if match: interval_mins = float(match.group(1))

        is_literal_hours = "timepoints" in main_jf.lower() or "96h" in exp_label.lower()
        tp_to_process = tp_sorted[:min_len]
        timepoints = data["timepoints"]
        raw_distributions[exp_label] = {m: [] for m in metrics_list}

        # BASELINE
        base_scalars = {m: None for m in metrics_list}
        if norm_mode in ['percent', 'absolute']:
            b_file = baseline_json_files[idx_f] if baseline_json_files else main_jf
            b_path = os.path.join(stats_dir, b_file)
            if os.path.exists(b_path):
                with open(b_path, "r") as f:
                    b_data = json.load(f)
                b_tps = natural_sort(list(b_data["timepoints"].keys()))
                
                if baseline_from_start:
                    b_eval_tps = b_tps[:baseline_calc_pts] if baseline_calc_pts else b_tps
                else:
                    b_eval_tps = b_tps[-baseline_calc_pts:] if baseline_calc_pts else b_tps
                    
                b_accum = {m: [] for m in metrics_list}
                area_mult = (pixel_size_um ** 2) if pixel_size_um is not None else 1.0
                
                for bt in b_eval_tps:
                    bt_raw = b_data["timepoints"][bt]["all_values"]
                    br_area = bt_raw.get("mean_area", [])
                    br_sol = bt_raw.get("mean_solidity", [])
                    br_ecc = bt_raw.get("mean_eccentricity", [])
                    br_ori = bt_raw.get("mean_orientation", [])
                    br_junc = bt_raw.get("mean_junction_fraction", [])
                    
                    for i in range(len(br_ecc)):
                        ecc_val = br_ecc[i]
                        area_val = br_area[i] * area_mult 
                        if min_ecc is not None and ecc_val < min_ecc: continue
                        if max_ecc is not None and ecc_val > max_ecc: continue
                        if min_area is not None and area_val < min_area: continue
                        if max_area is not None and area_val > max_area: continue
                        
                        b_accum["mean_area"].append(area_val)
                        b_accum["mean_solidity"].append(br_sol[i])
                        b_accum["mean_eccentricity"].append(br_ecc[i])
                        b_accum["mean_orientation"].append(br_ori[i])
                        if i < len(br_junc): b_accum["mean_junction_fraction"].append(br_junc[i])
                
                for m in metrics_list:
                    if b_accum[m]: base_scalars[m] = np.nanmean(b_accum[m]) 

        exp_time_data = []
        for idx, t in enumerate(tp_to_process):
            if is_literal_hours:
                try: real_time_hours = float(t.split('_')[0])
                except: real_time_hours = idx * (interval_mins / 60.0)
            else:
                real_time_hours = idx * (interval_mins / 60.0)
            
            tp_data = timepoints[t]
            raw_vals = tp_data["all_values"]
            r_area = raw_vals.get("mean_area", [])
            r_sol = raw_vals.get("mean_solidity", [])
            r_ecc = raw_vals.get("mean_eccentricity", [])
            r_ori = raw_vals.get("mean_orientation", [])
            r_junc = raw_vals.get("mean_junction_fraction", []) 
            
            filt_area, filt_sol, filt_ecc, filt_ori, filt_junc = [], [], [], [], []
            area_multiplier = (pixel_size_um ** 2) if pixel_size_um is not None else 1.0
            
            if len(r_ecc) > 0:
                for i in range(len(r_ecc)):
                    ecc_val = r_ecc[i]
                    area_val = r_area[i] * area_multiplier 
                    if min_ecc is not None and ecc_val < min_ecc: continue
                    if max_ecc is not None and ecc_val > max_ecc: continue
                    if min_area is not None and area_val < min_area: continue
                    if max_area is not None and area_val > max_area: continue
                    
                    filt_area.append(area_val)
                    filt_sol.append(r_sol[i])
                    filt_ecc.append(r_ecc[i])
                    filt_ori.append(r_ori[i])
                    if i < len(r_junc): filt_junc.append(r_junc[i])
                    
            exp_time_data.append({
                "idx": idx, "real_time": real_time_hours,
                "filt_area": filt_area, "filt_sol": filt_sol,
                "filt_ecc": filt_ecc, "filt_ori": filt_ori, "filt_junc": filt_junc 
            })

        for tp in exp_time_data:
            idx = tp["idx"]
            real_time_hours = tp["real_time"]
            raw_m_area = np.mean(tp["filt_area"]) if tp["filt_area"] else np.nan
            raw_m_sol = np.mean(tp["filt_sol"]) if tp["filt_sol"] else np.nan
            raw_m_ecc = np.mean(tp["filt_ecc"]) if tp["filt_ecc"] else np.nan
            raw_m_ori = np.mean(tp["filt_ori"]) if tp["filt_ori"] else np.nan
            raw_m_junc = np.mean(tp["filt_junc"]) if tp["filt_junc"] else np.nan

            normalized_means = {}
            for m, raw_val in zip(metrics_list, [raw_m_area, raw_m_sol, raw_m_ecc, raw_m_ori, raw_m_junc]):
                b_val = base_scalars[m]
                if pd.isna(raw_val): normalized_means[m] = np.nan
                elif norm_mode == 'percent' and b_val is not None and b_val != 0:
                    normalized_means[m] = ((raw_val - b_val) / b_val) * 100.0
                elif norm_mode == 'absolute' and b_val is not None:
                    normalized_means[m] = raw_val - b_val
                else:
                    normalized_means[m] = raw_val
            
            all_data.append({
                "experiment": exp_label, "real_time_hours": real_time_hours, 
                "mean_area": normalized_means["mean_area"],
                "mean_solidity": normalized_means["mean_solidity"],
                "mean_eccentricity": normalized_means["mean_eccentricity"],
                "mean_orientation": normalized_means["mean_orientation"],
                "mean_junction_fraction": normalized_means["mean_junction_fraction"] 
            })

            if ks_eval_points is None or idx >= (min_len - ks_eval_points):
                for metric in metrics_list:
                    val = normalized_means[metric]
                    if not np.isnan(val):
                        raw_distributions[exp_label][metric].append(val)

    df = pd.DataFrame(all_data)
    df = df.sort_values(["experiment", "real_time_hours"]) 
    experiment_names = list(raw_distributions.keys())

    max_time_in_plot = df["real_time_hours"].max() if not df.empty else 0
    use_minutes = (max_time_in_plot <= 1.05 and max_time_in_plot > 0)
    time_multiplier = 60.0 if use_minutes else 1.0
    x_axis_label = "Time (minutes)" if use_minutes else "Time (hours)"

    filter_suffix = ""
    if min_ecc is not None: filter_suffix += f"_minEcc{min_ecc}"
    if max_ecc is not None: filter_suffix += f"_maxEcc{max_ecc}"
    if min_area is not None: filter_suffix += f"_minArea{min_area}"
    if max_area is not None: filter_suffix += f"_maxArea{max_area}"
    if norm_mode == 'percent': filter_suffix += "_pct"
    if norm_mode == 'absolute': filter_suffix += "_abs"

    for metric in metrics_list:
        plt.figure(figsize=(12, 8)) 
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False) 
        
        for exp in experiment_names:
            df_exp = df[df["experiment"] == exp].sort_values("real_time_hours")
            x_vals = df_exp["real_time_hours"].values * time_multiplier 
            y_vals = df_exp[metric].values
            mask = ~np.isnan(y_vals)
            x_clean, y_clean = x_vals[mask], y_vals[mask]
            
            if len(x_clean) > 1:
                try:
                    y_smooth = gaussian_filter1d(y_clean, sigma=smoothing_sigma)
                    p = plt.plot(x_clean, y_smooth, label=exp, linewidth=2, zorder=2)
                    color = p[0].get_color()
                    plt.plot(x_clean, y_clean, "o", color=color, markersize=5, markeredgecolor='black', markeredgewidth=0.8, zorder=3)
                except:
                    plt.plot(x_vals, y_vals, marker="o", label=exp, markeredgecolor='black', markeredgewidth=0.8)
            else:
                plt.plot(x_vals, y_vals, marker="o", label=exp, markeredgecolor='black', markeredgewidth=0.8)

        unique_x = np.sort(df["real_time_hours"].unique()) * time_multiplier
        if len(unique_x) <= 12: plt.xticks(unique_x)
        else: ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
            
        ax.tick_params(axis='both', labelsize=12)
        
        base_metric_name = metric.replace("mean_", "").replace("_", " ").title()
        if norm_mode == 'percent': ylabel_text = f"Change in {base_metric_name} (%)"
        elif norm_mode == 'absolute': ylabel_text = f"Δ {base_metric_name} (Absolute)"
        else:
            if metric == "mean_area": ylabel_text = "Cell Area (μm²)" if pixel_size_um is not None else "Cell Area (pixels)"
            elif metric == "mean_junction_fraction": ylabel_text = "Junction Fraction (0 to 1)"
            else: ylabel_text = base_metric_name
                
        plt.ylabel(ylabel_text, fontsize=14, fontweight='bold')
        plt.xlabel(x_axis_label, fontsize=14, fontweight='bold') 
        plt.title(f"{base_metric_name} Distribution Evolution", fontsize=18, fontweight='bold')

        if group_mapping is not None:
            distributions_to_test = {}
            for exp in experiment_names:
                grp = group_mapping.get(exp, exp)
                if grp not in distributions_to_test:
                    distributions_to_test[grp] = {m: [] for m in metrics_list}
                distributions_to_test[grp][metric].extend(raw_distributions[exp][metric])
            test_names_grouped = list(distributions_to_test.keys())
        else:
            test_names_grouped = experiment_names

        # CÁLCULO ESTADÍSTICO (SIN IMPRIMIR EN PANTALLA)
        # 1. Individual
        if len(experiment_names) >= 2:
            for exp1, exp2 in combinations(experiment_names, 2):
                vals1 = raw_distributions[exp1][metric]
                vals2 = raw_distributions[exp2][metric]
                if len(vals1) > 0 and len(vals2) > 0:
                    stat, pval = mannwhitneyu(vals1, vals2, alternative='two-sided')
                    n1, n2 = len(vals1), len(vals2)
                    cliff_d = abs((2 * stat) / (n1 * n2) - 1)
                    
                    global_stats_list.append({
                        "Experiment": save_name,
                        "Metric": base_metric_name,
                        "Type": "Individual",
                        "Group A": exp1,
                        "Group B": exp2,
                        "Statistic (U/V)": round(stat, 3),
                        "p-value": pval,
                        "Cliff's Delta": round(cliff_d, 2)
                    })
                    
        # 2. Grouped
        if group_mapping is not None and len(test_names_grouped) >= 2:
            for grp1, grp2 in combinations(test_names_grouped, 2):
                vals1 = distributions_to_test[grp1][metric]
                vals2 = distributions_to_test[grp2][metric]
                if len(vals1) > 0 and len(vals2) > 0:
                    stat, pval = mannwhitneyu(vals1, vals2, alternative='two-sided')
                    n1, n2 = len(vals1), len(vals2)
                    cliff_d = abs((2 * stat) / (n1 * n2) - 1)
                    
                    global_stats_list.append({
                        "Experiment": save_name,
                        "Metric": base_metric_name,
                        "Type": "Grouped",
                        "Group A": grp1,
                        "Group B": grp2,
                        "Statistic (U/V)": round(stat, 3),
                        "p-value": pval,
                        "Cliff's Delta": round(cliff_d, 2)
                    })

        if norm_mode in ['percent', 'absolute']:
            plt.axhline(0, color='gray', linestyle='--', alpha=0.7)

        # LEYENDA Y GUARDADO (AHORA SIN EL CUADRO DE TEXTO AL LADO)
        plt.legend(title="Condition", loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   fancybox=False, shadow=False, ncol=3, frameon=False, fontsize=12, title_fontsize=13)
        
        if save_dir is not None:
            metric_folder = os.path.join(save_dir, metric)
            os.makedirs(metric_folder, exist_ok=True)
            filename = f"{save_name}_{metric}{filter_suffix}.jpg"
            save_path = os.path.join(metric_folder, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved CLEAN plot: {save_path}")
            
        plt.show()

def plot_polar_orientation(stats_dir, json_files=None, save_dir=None, save_name="angles", min_ecc=None, max_ecc=None, 
                           label_mapping=None, ks_eval_points=None, aggregate_by_frame=False, group_mapping=None):
    
    global global_stats_list
    
    if json_files is None:
        json_files = [f for f in os.listdir(stats_dir) if f.endswith(".json")]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    loaded_experiments = []
    min_len = float('inf')
    
    for jf in json_files:
        path = os.path.join(stats_dir, jf)
        with open(path, "r") as f:
            data = json.load(f)
            folder_name = data["folder"]
            tp_sorted = natural_sort(list(data["timepoints"].keys()))
            display_label = label_mapping[jf] if label_mapping is not None and jf in label_mapping else folder_name
            if len(tp_sorted) < min_len: min_len = len(tp_sorted)
            loaded_experiments.append((data, tp_sorted, display_label))

    raw_angle_distributions = {}

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1) 
    ax.spines['polar'].set_linewidth(1.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=12)

    palette = sns.color_palette("bright", len(json_files))

    for idx, (data, tp_sorted, display_label) in enumerate(loaded_experiments):
        all_angles_for_plot = []
        frame_means_for_test = []
        timepoints = data["timepoints"]
        
        if ks_eval_points is not None:
            start_idx = max(0, min_len - ks_eval_points)
            tp_to_process = tp_sorted[start_idx:min_len]
        else:
            tp_to_process = tp_sorted[:min_len]

        try:
            for t in tp_to_process:
                angles = timepoints[t]["all_values"]["mean_orientation"]
                eccentricities = timepoints[t]["all_values"]["mean_eccentricity"]
                frame_valid_angles = []
                for ang, ecc in zip(angles, eccentricities):
                    if min_ecc is not None and ecc < min_ecc: continue
                    if max_ecc is not None and ecc > max_ecc: continue
                    frame_valid_angles.append(ang)
                    all_angles_for_plot.append(ang)
                
                if len(frame_valid_angles) > 0:
                    c_mean = circmean(frame_valid_angles, high=180, low=0)
                    frame_means_for_test.append(c_mean)

            if not all_angles_for_plot:
                raw_angle_distributions[display_label] = []
                continue

            angles_deg_plot = np.array(all_angles_for_plot)
            if aggregate_by_frame: raw_angle_distributions[display_label] = np.deg2rad(np.array(frame_means_for_test))
            else: raw_angle_distributions[display_label] = np.deg2rad(angles_deg_plot)

            angles_rad_plot = np.deg2rad(angles_deg_plot)
            angles_rad_symmetric = np.concatenate([angles_rad_plot, angles_rad_plot + np.pi])
            angles_rad_symmetric = angles_rad_symmetric % (2 * np.pi)

            bins_number = 36 
            counts, bin_edges = np.histogram(angles_rad_symmetric, bins=bins_number, range=(0, 2*np.pi), density=True)
            theta = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts = np.append(counts, counts[0])
            theta = np.append(theta, theta[0])

            ax.plot(theta, counts, color=palette[idx], linewidth=2.5, label=display_label)
            ax.fill(theta, counts, color=palette[idx], alpha=0.1)

        except Exception as e:
            print(f"Error processing: {e}")

    plt.title(f"Orientation Distribution", y=1.08, fontsize=18, fontweight='bold')

    experiment_names = list(raw_angle_distributions.keys())
    
    # 1. Individual
    if len(experiment_names) >= 2:
        for exp1, exp2 in combinations(experiment_names, 2):
            vals1 = raw_angle_distributions[exp1]
            vals2 = raw_angle_distributions[exp2]
            if len(vals1) > 0 and len(vals2) > 0:
                stat, pval = kuiper_two(vals1, vals2)
                global_stats_list.append({
                    "Experiment": save_name,
                    "Metric": "Orientation",
                    "Type": "Individual",
                    "Group A": exp1,
                    "Group B": exp2,
                    "Statistic (U/V)": round(stat, 3),
                    "p-value": pval,
                    "Cliff's Delta": "N/A (Kuiper)"
                })

    # 2. Grouped
    if group_mapping is not None:
        distributions_to_test = {}
        for exp in experiment_names:
            grp = group_mapping.get(exp, exp)
            if grp not in distributions_to_test:
                distributions_to_test[grp] = []
            distributions_to_test[grp].extend(raw_angle_distributions[exp])
        
        test_names_grouped = list(distributions_to_test.keys())
        if len(test_names_grouped) >= 2:
            for grp1, grp2 in combinations(test_names_grouped, 2):
                vals1 = distributions_to_test[grp1]
                vals2 = distributions_to_test[grp2]
                if len(vals1) > 0 and len(vals2) > 0:
                    stat, pval = kuiper_two(vals1, vals2)
                    global_stats_list.append({
                        "Experiment": save_name,
                        "Metric": "Orientation",
                        "Type": "Grouped",
                        "Group A": grp1,
                        "Group B": grp2,
                        "Statistic (U/V)": round(stat, 3),
                        "p-value": pval,
                        "Cliff's Delta": "N/A (Kuiper)"
                    })

    # DIBUJO LIMPIO SIN TEXTO
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=12, title_fontsize=13)
    lines, labels = plt.thetagrids(range(0, 360, 45), labels=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    
    if save_dir is not None:
        metric = "polar_lines"
        metric_folder = os.path.join(save_dir, metric)
        os.makedirs(metric_folder, exist_ok=True)
        suffix = ""
        if min_ecc is not None: suffix += f"_min{min_ecc}"
        if max_ecc is not None: suffix += f"_max{max_ecc}"
        filename = f"{save_name}_polar_lines{suffix}.jpg"
        save_path = os.path.join(metric_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved CLEAN polar plot: {save_path}")

    plt.show()

# -------------------------------------------------------------------------------------
# EJECUCIÓN ESTADÍSTICA Y GRÁFICAS
# -------------------------------------------------------------------------------------

stats_dir = "./segmentation_stats"
save_dir = "./plots"
fApplySelection = 1

lastTimePointsToEval = 1000
aggregateByFrame=True 


# EDTA test (Growth) ------------------------------------------------------------------------------
fDo = 1
if fDo == 1:
    json_files = ["CRU006_2025-07-17_15-04_CRU006 (OFFLINE).json"]
    labels = ["Growth"]
    labels = dict(zip(json_files, labels))
    
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "edta", label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "edta", min_ecc=0.85, label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "edta", max_ecc=0.85, label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="edta_growth", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="edta_growth", min_ecc=0.85, label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="edta_growth", max_ecc=0.85, label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)

fDo = 1
if fDo == 1:
    json_files = ["CRU006_2025-07-21_14-34_CRU006 - EDTA - 01Mm - EVERY5min (OFFLINE).json",
                  "CRU006_2025-07-22_14-33_CRU006 - EDTA - 5Mm - EVERY5min (OFFLINE).json",
                  "CRU006_2025-07-23_9-03_CRU006 - EDTA - 10Mm - EVERY5min (OFFLINE).json"]
    labels = ["EDTA 0.1 mM", "EDTA 5 mM", "EDTA 10 mM"]
    baseline_files = ["CRU006_2025-07-17_15-04_CRU006 (OFFLINE).json",
                      "CRU006_2025-07-17_15-04_CRU006 (OFFLINE).json",
                      "CRU006_2025-07-17_15-04_CRU006 (OFFLINE).json"]
    
    if fApplySelection == 1:
        indices = (0, 1, 2)
        json_files = [json_files[i] for i in indices]
        baseline_files = [baseline_files[i] for i in indices]
        labels = [labels[i] for i in indices]
        labels = dict(zip(json_files, labels))

    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "edta_vs_growth", label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame,
                                  baseline_json_files=json_files,
                                  baseline_calc_pts=5, baseline_from_start=True,
                                  norm_mode='percent')
    
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="edta", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="edta", min_ecc=0.85, label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="edta", max_ecc=0.85, label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)

fDo = 1
if fDo == 1:
    json_files = ["CRU002_CRU002 - CH1 - 0.json", 
                  "CRU003_CRU003 - CH1 - 0.json", 
                  "CRU004_CRU004 - CH1 - 0.json"]      
    labels = ["Healthy Exp 1", "BQL Exp 1", "BQL Exp 2"]     
    labels = dict(zip(json_files, labels))            
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "crecimiento", label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="crecimiento", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)

fDo = 1
if fDo == 1:
    json_files = ["CRU002_CRU002 - CH1 - 1 - BEFORE.json", 
                  "CRU003_CRU003 - CH2 - 1 - BEFORE.json", 
                  "CRU004_CRU004 - CH1 - 1 - BEFORE.json",
                  "CRU008_CRU008 - CH1 - 1 - BEFORE.json",
                  "CRU010_CRU010 - CH1 - 1 - BEFORE.json",
                  "CRU012_CRU012 - CH2 - 1 - BEFORE.json"]
    labels = ["Healthy Exp 1", "BQL Exp 1", "BQL Exp 2", "Healthy Exp 2", "Healthy Exp 3", "Healthy Exp 4"]
              
    if fApplySelection == 1:
        indices = (0, 1, 2, 3)
        json_files = [json_files[i] for i in indices]
        labels = [labels[i] for i in indices]
        
    labels = dict(zip(json_files, labels))                 
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "before", label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="before", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)

fDo = 1
if fDo == 1:
    json_files = ["CRU002_CRU002 - CH1 - 2 - TRIGGER - HLT.json", 
                  "CRU003_CRU003 - CH2 - 2 - TRIGGER - BQL.json", 
                  "CRU004_CRU004 - CH1 - 2 - TRIGGER - BQL.json",
                  "CRU008_CRU008 - CH1 - 2 - TRIGGER - HLT.json",
                  "CRU010_CRU010 - CH1 - 2 - TRIGGER - HLT.json",
                  "CRU012_CRU012 - CH2 - 2 - TRIGGER - HLT.json"]
    baseline_files = ["CRU002_CRU002 - CH1 - 1 - BEFORE.json", 
                      "CRU003_CRU003 - CH2 - 1 - BEFORE.json", 
                      "CRU004_CRU004 - CH1 - 1 - BEFORE.json",
                      "CRU008_CRU008 - CH1 - 1 - BEFORE.json",
                      "CRU010_CRU010 - CH1 - 1 - BEFORE.json",
                      "CRU012_CRU012 - CH2 - 1 - BEFORE.json"]
    labels = ["Healthy Exp 1", "BQL Exp 1", "BQL Exp 2", "Healthy Exp 2", "Healthy Exp 3", "Healthy Exp 4"]
    gm = {"Healthy Exp 1": "Healthy (Control)", "BQL Exp 1": "BQL (Infected)", "BQL Exp 2": "BQL (Infected)", "Healthy Exp 2": "Healthy (Control)", "Healthy Exp 3": "Healthy (Control)", "Healthy Exp 4": "Healthy (Control)"}
    
    if fApplySelection == 1:
        indices = (0, 1, 2, 3)
        json_files = [json_files[i] for i in indices]
        baseline_files = [baseline_files[i] for i in indices]
        labels = [labels[i] for i in indices]
        gm = {"Healthy Exp 1": "Healthy (Control)", "BQL Exp 1": "BQL (Infected)", "BQL Exp 2": "BQL (Infected)", "Healthy Exp 2": "Healthy (Control)"}

    labels = dict(zip(json_files, labels))                
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "trigger", label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame,
                                  group_mapping=gm, baseline_json_files=json_files, baseline_from_start=True,
                                  baseline_calc_pts=1, norm_mode='percent')
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="trigger", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm)

fDo = 1
if fDo == 1:
    json_files = ["CRU002_CRU002 - CH1 - 3 - AFTER.json", 
                  "CRU003_CRU003 - CH2 - 3 - AFTER.json", 
                  "CRU004_CRU004 - CH1 - 3 - AFTER.json",
                  "CRU008_CRU008 - CH1 - 3 - AFTER.json",
                  "CRU010_CRU010 - CH1 - 3 - AFTER.json",
                  "CRU012_CRU012 - CH2 - 3 - AFTER.json"]   
    baseline_files = ["CRU002_CRU002 - CH1 - 2 - TRIGGER - HLT.json", 
                      "CRU003_CRU003 - CH2 - 2 - TRIGGER - BQL.json", 
                      "CRU004_CRU004 - CH1 - 2 - TRIGGER - BQL.json",
                      "CRU008_CRU008 - CH1 - 2 - TRIGGER - HLT.json",
                      "CRU010_CRU010 - CH1 - 2 - TRIGGER - HLT.json",
                      "CRU012_CRU012 - CH2 - 2 - TRIGGER - HLT.json"]
    labels = ["Healthy Exp 1", "BQL Exp 1", "BQL Exp 2", "Healthy Exp 2", "Healthy Exp 3", "Healthy Exp 4"]
    gm = {"Healthy Exp 1": "Healthy (Control)", "BQL Exp 1": "BQL (Infected)", "BQL Exp 2": "BQL (Infected)", "Healthy Exp 2": "Healthy (Control)", "Healthy Exp 3": "Healthy (Control)", "Healthy Exp 4": "Healthy (Control)"}
    
    if fApplySelection == 1:
        indices = (0, 1, 2, 3)
        json_files = [json_files[i] for i in indices]
        labels = [labels[i] for i in indices]
        baseline_files = [baseline_files[i] for i in indices]
        gm = {"Healthy Exp 1": "Healthy (Control)", "BQL Exp 1": "BQL (Infected)", "BQL Exp 2": "BQL (Infected)", "Healthy Exp 2": "Healthy (Control)"}

    labels = dict(zip(json_files, labels))   
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "after_vs_trigger", 
                                  max_area=450, label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm, 
                                  baseline_json_files=json_files, baseline_from_start=True,
                                  baseline_calc_pts=5, norm_mode='percent')
    
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="after", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="after", min_ecc=0.85, label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="after", max_ecc=0.85, label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm)

fDo = 1
if fDo == 1:    
    json_files = ["CRU000_timepoints_0_4_24_48_72_96_CRU000.json"]
    labels = ["Growth (96h)"] 
    labels_dict = dict(zip(json_files, labels))
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name="cru000_96h", 
                                  label_mapping=labels_dict, ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="cru000_96h", 
                           label_mapping=labels_dict, ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame)

# AL FINALIZAR, EXPORTAMOS TODO EL REGISTRO A CSV
save_statistics_to_csv("statistical_results_table.csv")
print("\nProcesamiento estadístico completado.")




