#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenta imágenes .jpg con Cellpose usando un Grid Search adaptativo por frame,
guarda parámetros óptimos y estadísticas, genera heatmaps célula a célula 
y crea gráficos temporales/polares con test estadísticos.
"""

import os

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

sns.set(style="whitegrid")

# --- Carpetas raíz ---
root_dir = "./"
stats_output_dir = "./segmentation_stats/"
overlay_output_dir = "./segmentation_overlay/"
masks_output_dir = "./segmentation_masks/" # --- NUEVA CARPETA PARA MÁSCARAS COMPRIMIDAS ---
os.makedirs(stats_output_dir, exist_ok=True)
os.makedirs(overlay_output_dir, exist_ok=True)
os.makedirs(masks_output_dir, exist_ok=True)

# --------------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------------------


def compute_mask_stats(label_mask):
    stats = []
    
    # --- CÁLCULO VECTORIZADO DE TIGHT JUNCTIONS ---
    # Desplazamos la máscara en las 4 direcciones para comparar vecinos
    up = np.roll(label_mask, -1, axis=0)
    down = np.roll(label_mask, 1, axis=0)
    left = np.roll(label_mask, -1, axis=1)
    right = np.roll(label_mask, 1, axis=1)

    # Evitamos falsos positivos en los bordes absolutos de la imagen
    up[-1, :] = 0
    down[0, :] = 0
    left[:, -1] = 0
    right[:, 0] = 0

    # Píxeles de perímetro: pertenecen a una célula y al menos un vecino es distinto a ella
    is_perim = (label_mask > 0) & (
        (up != label_mask) | (down != label_mask) |
        (left != label_mask) | (right != label_mask)
    )

    # Píxeles de unión (Junction): pertenecen a una célula y al menos un vecino es OTRA célula
    is_junct = (label_mask > 0) & (
        ((up > 0) & (up != label_mask)) |
        ((down > 0) & (down != label_mask)) |
        ((left > 0) & (left != label_mask)) |
        ((right > 0) & (right != label_mask))
    )

    # Contamos cuántos píxeles de perímetro y unión tiene cada célula (usando su ID como índice)
    perim_counts = np.bincount(label_mask[is_perim])
    junct_counts = np.bincount(label_mask[is_junct])
    # ----------------------------------------------

    for prop in regionprops(label_mask):
        if prop.area < 10:  # filtro mínimo
            continue
            
        lbl = prop.label
        
        # Calcular la fracción de perímetro compartido (de 0.0 a 1.0)
        try:
            p_count = perim_counts[lbl]
            j_count = junct_counts[lbl]
            junction_fraction = j_count / p_count if p_count > 0 else 0.0
        except IndexError:
            junction_fraction = 0.0

        stats.append({
            "label": lbl,
            "area": float(prop.area),
            "solidity": float(prop.solidity),
            "eccentricity": float(prop.eccentricity),
            "orientation": float(np.degrees(prop.orientation) % 180),
            "junction_fraction": float(junction_fraction) # NUEVA MÉTRICA
        })
    return stats







def save_segmentation_overlay(img_gray, mask, save_path, thickness=2):
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    for label_id in np.unique(mask):
        if label_id == 0: continue
        single_mask = (mask==label_id).astype(np.uint8)
        contours,_ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.drawContours(img_rgb, contours, -1, color, thickness)
    Image.fromarray(img_rgb).save(save_path)

def save_metric_colormap_overlay(img_gray, label_mask, stats, metric_name, save_path, vmin=None, vmax=None, cmap_name='turbo', alpha=0.7):
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

    colored_mask = np.zeros((*label_mask.shape, 4))
    cmap = plt.get_cmap(cmap_name)
    
    if vmin is None: vmin = np.nanmin(val_array)
    if vmax is None: vmax = np.nanmax(val_array)
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    colored_mask[valid_mask] = cmap(norm(metric_img[valid_mask]))
    colored_mask[valid_mask, 3] = alpha

    ax.imshow(colored_mask)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name.replace("_", " ").title(), fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)

def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# --------------------------------------------------------------------------------
# PROCESAMIENTO DE IMÁGENES (ADAPTIVE GRID SEARCH)
# --------------------------------------------------------------------------------


# CLAHE(clip=0.05, kernel=(64, 64)) | Cellpose(prob=-10, flow=0.99) ==> 3440 células
# CLAHE(clip=0.05, kernel=(64, 64)) | Cellpose(prob=-10, flow=0.99) ==> 3939 células
# CLAHE(clip=0.01, kernel=(128, 128)) | Cellpose(prob=-5, flow=0.99) ==> 2052 células
# CLAHE(clip=0.01, kernel=(128, 128)) | Cellpose(prob=-10, flow=0.99) ==> 2303 células  
# CLAHE(clip=0.01, kernel=(128, 128)) | Cellpose(prob=-10, flow=0.99) ==> 2284 células
# CLAHE(clip=0.01, kernel=(64, 64)) | Cellpose(prob=-10, flow=0.99) ==> 2196 células

# CLAHE(clip=0.1, kernel=(32, 32)) | Cellpose(prob=-20, flow=0.99) ==> 3458 células
# CLAHE(clip=0.05, kernel=(32, 32)) | Cellpose(prob=-20, flow=0.99) ==> 4176 células
# CLAHE(clip=0.01, kernel=(128, 128)) | Cellpose(prob=-20, flow=0.99) ==> 2052 células
# CLAHE(clip=0.01, kernel=(256, 256)) | Cellpose(prob=-20, flow=0.99) ==> 2315 células

# "0_h.jpg": {"": 0.1,"": [32,32],"": -30,"flow_threshold": 0.99,"": 3458},
# "4_h.jpg": {"": 0.05,"": [32,32],"": -30,"flow_threshold": 0.99,"": 4176},
# "24_h.jpg": {"": 0.01,"": [128,128],"": -30,"flow_threshold": 0.99,"": 2052}
# "48_h.jpg": {"": 0.01,"": [256,256],"": -30,"flow_threshold": 0.99,"": 2315}
# "72_h.jpg": {"": 0.01,"": [512,512],"": -30,"flow_threshold": 0.99,"": 2296}
# "96_h.jpg": {"": 0.01,"": [32,32],"": -30,"flow_threshold": 0.99,"": 2275}


# CLAHE(clip=0.05, kernel=(16, 16)) | Cellpose(prob=-1000, flow=0.99) ==> 3489 células
# CLAHE(clip=0.05, kernel=(16, 16)) | Cellpose(prob=-1000, flow=0.99) ==> 4391 células
# CLAHE(clip=0.01, kernel=(1024, 1024)) | Cellpose(prob=-1000, flow=0.99) ==> 2061 células
# CLAHE(clip=0.01, kernel=(1024, 1024)) | Cellpose(prob=-1000, flow=0.99) ==> 2358 células
# CLAHE(clip=0.01, kernel=(1024, 1024)) | Cellpose(prob=-1000, flow=0.99) ==> 2347 células
# CLAHE(clip=0.01, kernel=(32, 32)) | Cellpose(prob=-1000, flow=0.99) ==> 2275 células


# 0_h.jpg: clip_limit=0.05, kernel_size=[16,16], =-1000000, =0.99, valid_cells_found=3489
# 4_h.jpg: clip_limit=0.05, kernel_size=[16,16], =-1000000, =0.99, valid_cells_found=4391
# 24_h.jpg: clip_limit=0.01, kernel_size=[1024,1024], =-1000000, =0.99, valid_cells_found=2061
# 48_h.jpg: clip_limit=0.01, kernel_size=[1024,1024], =-1000000, =0.99, valid_cells_found=2358
# 72_h.jpg: clip_limit=0.01, kernel_size=[1024,1024], =-1000000, =0.99, valid_cells_found=2347
# 96_h.jpg: clip_limit=0.01, kernel_size=[32,32], =-1000000, =0.99, valid_cells_found=2275


# 0_h.jpg: clip_limit=0.05, kernel_size=[16,16], cellprob_threshold=-1000000, flow_threshold=0.99, diameter=null, valid_cells_found=3572
# 4_h.jpg: clip_limit=0.05, kernel_size=[16,16], cellprob_threshold=-1000000, flow_threshold=0.99, diameter=20, valid_cells_found=4448
# 24_h.jpg: clip_limit=0.01, kernel_size=[1024,1024], cellprob_threshold=-1000000, flow_threshold=0.99, diameter=null, valid_cells_found=2518
# 48_h.jpg: clip_limit=0.01, kernel_size=[1024,1024], cellprob_threshold=-1000000, flow_threshold=0.99, diameter=35, valid_cells_found=2868
# 72_h.jpg: clip_limit=0.01, kernel_size=[1024,1024], cellprob_threshold=-1000000, flow_threshold=0.99, diameter=35, valid_cells_found=3036
# 96_h.jpg: clip_limit=0.01, kernel_size=[256,256], cellprob_threshold=-1000000, flow_threshold=0.99, diameter=35, valid_cells_found=3334

# 0_h.jpg: clip_limit=0.05, kernel_size=(16,16), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=30, valid_cells_found=3572
# 4_h.jpg: clip_limit=0.05, kernel_size=(16,16), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=50, valid_cells_found=4819
# 24_h.jpg: clip_limit=0.05, kernel_size=(256,256), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=50, valid_cells_found=4045
# 48_h.jpg: clip_limit=0.01, kernel_size=(16,16), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=50, valid_cells_found=3941
# 72_h.jpg: clip_limit=0.05, kernel_size=(128,128), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=50, valid_cells_found=5171
# 96_h.jpg: clip_limit=0.05, kernel_size=(1024,1024), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=50, valid_cells_found=6419


# 0_h.jpg: clip_limit=0.05, kernel_size=(8,8), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=30, valid_cells_found=3654
# 4_h.jpg: clip_limit=0.05, kernel_size=(32,32), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=60, valid_cells_found=4993
# 24_h.jpg: clip_limit=0.01, kernel_size=(8,8), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=70, valid_cells_found=4838
# 48_h.jpg: clip_limit=0.05, kernel_size=(512,512), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=70, valid_cells_found=5643
# 72_h.jpg: clip_limit=0.05, kernel_size=(512,512), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=60, valid_cells_found=5974
# 96_h.jpg: clip_limit=0.05, kernel_size=(512,512), cellprob_threshold=-1e6, flow_threshold=0.99, diameter=60, valid_cells_found=6800




def process_image_folder(folder_path, relative_name, fTestImages=0, fForceGridSearch=0):
    interval = 0
    
    print(f"\nProcesando carpeta: {relative_name}")
    timepoints_data = {}
    l_files = natural_sort([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])
    
    
    # folder_path = "./CRU000_growth_cell_seeding_and_day_1/CRU004 - CH1 - 0"
    # l_files = natural_sort([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])

    
    
    timestamps = [int(f.split('_')[0]) for f in l_files]
    intervals = [(timestamps[i+1] - timestamps[i])/60000 for i in range(len(timestamps)-1)]

    if len(set(intervals)) == 1:
        print(f"All intervals are equal: {intervals[0]:.2f} minutes")
        interval = int(np.round(intervals[0]))
        if interval == 0:
            interval = 1
    else:
        print("Intervals are not uniform!")
        interval = 1
        
        
    print("Interval is " + str(interval))
    
    
    
    # Parámetros de validación morfológica bruta
    min_size = 200
    max_size = 3000 
    
    if fTestImages > 0 and fTestImages < len(l_files):
        indices = np.linspace(0, len(l_files) - 1, int(fTestImages), dtype=int)
        l_files = [l_files[i] for i in indices]

    ind_f = 0
    len_f = str(len(l_files))
    
    # --- CONFIGURACIÓN DEL GRID SEARCH ---
    grid_clip_limits = [0.05] # 1 Nivel de contraste
    grid_kernel_sizes = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256)] # 5 Tamaños de ventana para iluminación
    grid_cellprob = [-1000000] # 1 Nivel de permisividad (rescate profundo)
    grid_flow = [0.99] # 1 Nivel de tolerancia de forma
    grid_diameters = [30, 50, 60, 70] # 6 Escalas
    
    # Ahora cruzamos 5 variables
    param_grid = list(itertools.product(grid_clip_limits, grid_kernel_sizes, grid_cellprob, grid_flow, grid_diameters))
    
    # Cargar historial de parámetros óptimos si existe
    opt_params_path = os.path.join(stats_output_dir, f"{relative_name.replace('/','_')}_opt_params.json")
    saved_opt_params = {}
    if os.path.exists(opt_params_path):
        with open(opt_params_path, "r") as f:
            saved_opt_params = json.load(f)
            print(f"Historial de parámetros cargado ({len(saved_opt_params)} frames cacheados).")

    for file in l_files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert("L")
        img_np = np.array(img)
        
        ind_f += 1
        print(f"\nProcessing image # {ind_f} out of {len_f} : {file}")
        
        best_num_cells = -1
        best_masks = None
        best_img_proc = None
        best_params_dict = {}

        if file in saved_opt_params and fForceGridSearch == 0:
            print("  -> Usando parámetros cacheados...")
            p = saved_opt_params[file]
            cl = p['clip_limit']
            ks = tuple(p['kernel_size']) # Recuperamos el kernel del JSON
            cpt = p['cellprob_threshold']
            ft = p['flow_threshold']
            diam = p['diameter']
            
            img_proc = exposure.equalize_adapthist(img_np, clip_limit=cl, kernel_size=ks)
            img_proc = img_as_ubyte(img_proc)
            try:
                masks, _, _ = model.eval(img_proc, diameter=diam, flow_threshold=ft, cellprob_threshold=cpt, 
                                         min_size=15, niter=200, normalize={"percentile":[1,99], "normalize":True})
                best_masks = masks
                best_img_proc = img_proc
            except Exception as e:
                print(f"Error evaluating cached params: {e}")
                continue
                
        else:
            if fForceGridSearch == 1 and file in saved_opt_params:
                print(f"  -> Forzando recálculo (ignorando caché) - {len(param_grid)} combinaciones...")
            else:
                print(f"  -> Buscando parámetros óptimos ({len(param_grid)} combinaciones)...")
                
            # Desempaquetamos las 5 variables de la cuadrícula
            for cl, ks, cpt, ft, diam in param_grid:
                try:
                    img_proc = exposure.equalize_adapthist(img_np, clip_limit=cl, kernel_size=ks)
                    img_proc = img_as_ubyte(img_proc)
                    
                    masks, _, _ = model.eval(img_proc, diameter=diam, flow_threshold=ft, cellprob_threshold=cpt, 
                                             min_size=15, niter=200, normalize={"percentile":[1,99], "normalize":True})
                    
                    if np.max(masks) == 0:
                        valid_cells = 0
                    else:
                        props = measure.regionprops(masks)
                        areas = np.array([prop.area for prop in props])
                        valid_cells = np.sum((areas >= min_size) & (areas <= max_size))
                    
                    
                    if valid_cells > best_num_cells:
                        # --- PRINT DE MONITORIZACIÓN ---
                        print(f"    [Prueba] Clip={cl} | Kernel={ks[0]} | Prob={cpt} | Flow={ft} | Diam={diam} ==> {valid_cells} células")
                            
                        best_num_cells = valid_cells
                        best_masks = masks.copy()
                        best_img_proc = img_proc.copy()
                        best_params_dict = {
                            "clip_limit": cl,
                            "kernel_size": list(ks),
                            "cellprob_threshold": cpt,
                            "flow_threshold": ft,
                            "diameter": diam,
                            "valid_cells_found": int(valid_cells)
                        }
                except Exception as e:
                    print(f"    [Error] Falló: Clip={cl}, Kernel={ks}, Prob={cpt}, Flow={ft}, Diam={diam}. Motivo: {e}")
                    pass
            
            if best_params_dict:
                saved_opt_params[file] = best_params_dict
                with open(opt_params_path, "w") as f:
                    json.dump(saved_opt_params, f, indent=2)
                print(f"  -> ¡ÓPTIMO ENCONTRADO! Ganador: {best_params_dict['valid_cells_found']} células (Kernel: {best_params_dict['kernel_size'][0]}, Diam: {best_params_dict['diameter']}).")
            else:
                print("  -> Fallo crítico: 0 células encontradas en todas las pruebas.")
                continue
    

        # --- Extracción de Estadísticas con los mejores parámetros ---
        try:
            if best_masks is None or np.max(best_masks) == 0:
                continue

            mask_ids = best_masks.copy()
            props = measure.regionprops(mask_ids)
            areas = np.array([prop.area for prop in props])
            labels = np.array([prop.label for prop in props])
            
            keep_labels = labels[(areas >= min_size) & (areas <= max_size)]
            filtered_mask = np.isin(mask_ids, keep_labels) * mask_ids
            filtered_mask = measure.label(filtered_mask)
            label_mask = filtered_mask.astype(np.int32)
            
            # --- NUEVO: GUARDAR MÁSCARA COMPRIMIDA (.npz) PARA LA EROSIÓN EN PLOTS ---
            raw_masks_folder = os.path.join(masks_output_dir, relative_name.replace("/", "_"))
            os.makedirs(raw_masks_folder, exist_ok=True)
            mask_save_path = os.path.join(raw_masks_folder, file.replace(".jpg", ".npz"))
            np.savez_compressed(mask_save_path, mask=label_mask)
            # --------------------------------------------------------------------------

            stats = compute_mask_stats(label_mask)

            # 1. Overlay estándar
            overlay_folder = os.path.join(overlay_output_dir, relative_name.replace("/","_"))
            os.makedirs(overlay_folder, exist_ok=True)
            overlay_path = os.path.join(overlay_folder, file)
            save_segmentation_overlay(best_img_proc, label_mask, overlay_path)

            # 2. Guardar Colormaps (Heatmaps célula a célula)
            colormaps_dir = os.path.join(overlay_folder, "colormaps")
            os.makedirs(colormaps_dir, exist_ok=True)

            save_metric_colormap_overlay(best_img_proc, label_mask, stats, "area", 
                                         os.path.join(colormaps_dir, f"area_{file}"), 
                                         vmin=200, vmax=2000, cmap_name='turbo')
            
            save_metric_colormap_overlay(best_img_proc, label_mask, stats, "eccentricity", 
                                         os.path.join(colormaps_dir, f"eccentricity_{file}"), 
                                         vmin=0.0, vmax=1.0, cmap_name='plasma')
            
            save_metric_colormap_overlay(best_img_proc, label_mask, stats, "solidity", 
                                         os.path.join(colormaps_dir, f"solidity_{file}"), 
                                         vmin=0.8, vmax=1.0, cmap_name='viridis')
            
            save_metric_colormap_overlay(best_img_proc, label_mask, stats, "orientation", 
                                         os.path.join(colormaps_dir, f"orientation_{file}"), 
                                         vmin=0.0, vmax=180.0, cmap_name='hsv')

            save_metric_colormap_overlay(best_img_proc, label_mask, stats, "orientation", 
                                         os.path.join(colormaps_dir, f"orientation_{file}"), 
                                         vmin=0.0, vmax=180.0, cmap_name='hsv')
                                         
            # --- NUEVO COLORMAP PARA TIGHT JUNCTIONS ---
            save_metric_colormap_overlay(best_img_proc, label_mask, stats, "junction_fraction", 
                                         os.path.join(colormaps_dir, f"junction_{file}"), 
                                         vmin=0.0, vmax=1.0, cmap_name='magma')

            # --- Guardar estadísticas (ACTUALIZADO CON JUNCTIONS) ---
            stats_array = np.array([[s["area"], s["solidity"], s["eccentricity"], s["orientation"], s["junction_fraction"]] for s in stats])
            
            if len(stats_array) > 0:
                mean_vals = stats_array.mean(axis=0)
                timepoints_data[file] = {
                    "num_cells": len(stats),
                    "mean_area": float(mean_vals[0]),
                    "mean_solidity": float(mean_vals[1]),
                    "mean_eccentricity": float(mean_vals[2]),
                    "mean_orientation": float(mean_vals[3]),
                    "mean_junction_fraction": float(mean_vals[4]), # NUEVA MEDIA
                    "all_values": {
                        "mean_area": [s["area"] for s in stats],
                        "mean_solidity": [s["solidity"] for s in stats],
                        "mean_eccentricity": [s["eccentricity"] for s in stats],
                        "mean_orientation": [s["orientation"] for s in stats],
                        "mean_junction_fraction": [s["junction_fraction"] for s in stats] # NUEVO ARRAY
                    }
                }
        except Exception as e:
            print(f"Error procesando estadísticas finales de {file}: {e}")



    if timepoints_data:
        output_path = os.path.join(stats_output_dir, f"{relative_name.replace('/','_')}.json")
        with open(output_path, "w") as f:
            json.dump({
                "folder": relative_name,
                "interval": interval,
                "timepoints": timepoints_data
            }, f, indent=2)
        print(f"Guardado: {output_path}")
    else:
        print(f"No se detectaron células válidas en toda la carpeta {relative_name}")


# --------------------------------------------------------------------------------
# EJECUCIÓN DE EXTRACCIÓN (CELLPOSE)
# --------------------------------------------------------------------------------

fCompute = 1
fForceGridSearch = 0  # 0 = Usa caché si existe; 1 = Ignora caché y recalcula siempre
fTestFolder = 1000
fTestImages = 0
# foldersToCompute = ["CRU008"]
# foldersToCompute = ["CRU000_timepoints_0_4_24_48_72_96"]

foldersToCompute = [
                    "CRU000_timepoints_0_4_24_48_72_96",
                    "CRU000_growth_cell_seeding_and_day_1",
                    "CRU002",
                    "CRU003",
                    "CRU004",
                    "CRU006",
                    "CRU008",
                    "CRU010",
                    "CRU012"
                    ]


fRemoveOverlays = False
if fRemoveOverlays == True:
    folder_path = "./segmentation_overlay"
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

if fCompute==1:
    
    # 1. Los Generalistas (Sirven para todo)
    # cyto3: El estándar actual. El mejor equilibrio entre velocidad y precisión. *
    # cpsam: El más potente, pero el más lento. *
    # cyto2: El rey de la generación anterior. Muy robusto.
    # cyto: El abuelo fundador. (Ya casi no se usa).
    # nuclei: Exclusivo para segmentar núcleos (ideal para canales DAPI/Hoechst).
        
    # 2. Los Especialistas de Cellpose 3.0 (Tus mejores amigos)
    # livecell_cp3: ¡Ojo a este! Especializado en células vivas sin tinción (contraste de fase / campo claro). 
    # Acostumbrado a ver células cambiando de forma y apiñándose. *
    # tissuenet_cp3: Especializado en cortes de tejido histológico súper densos.
    
    # 3. Los Especialistas en Microbiología
    # bact_phase_cp3: Bacterias en contraste de fase.
    # bact_fluor_cp3: Bacterias con fluorescencia.
    # yeast_ph_cp3: Levaduras en contraste de fase.
    # yeast_bf_cp3: Levaduras en campo claro.
    # deepbacs_cp3: Una red especializada en bacterias muy difíciles o con mucho ruido.

    # model = models.CellposeModel(gpu=True, pretrained_model='/home/ccastilla/.cellpose/models/cytotorch_3')
    # model = models.CellposeModel(gpu=True, model_type='cyto3')
    # model = models.Cellpose(gpu=True, model_type='cpsam')
    # model = models.Cellpose(gpu=True, model_type='livecell_cp3')

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    
    contFolder = 0
    for main_folder in sorted(os.listdir(root_dir), reverse=False):
        main_path = os.path.join(root_dir, main_folder)
        if not (os.path.isdir(main_path) and main_folder.startswith("CRU") and main_folder in foldersToCompute):
            continue
        
        print("Computing folder " + main_folder + " !")
        for subfolder in sorted(os.listdir(main_path), reverse=True):
            subfolder_path = os.path.join(main_path, subfolder)
            if os.path.isdir(subfolder_path):
                rel_name = os.path.relpath(subfolder_path, root_dir)
                process_image_folder(subfolder_path, rel_name, fTestImages, fForceGridSearch)
                
                contFolder += 1
                if contFolder==fTestFolder: break
        if contFolder==fTestFolder: break

    print("\nProcesamiento de extracción completado.")


# -------------------------------------------------------------------------------------
# NUEVA FUNCIÓN PARA APLICAR EROSIÓN REAL DESDE LA MÁSCARA EN DISCO
# -------------------------------------------------------------------------------------
def get_eroded_stats_from_disk(folder_rel_name, filename, erosion_radius):
    mask_path = os.path.join(masks_output_dir, folder_rel_name.replace("/", "_"), filename.replace(".jpg", ".npz"))
    if not os.path.exists(mask_path):
        return None
        
    try:
        with np.load(mask_path) as data:
            label_mask = data['mask']
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_radius + 1, 2 * erosion_radius + 1))
        eroded_mask = np.zeros_like(label_mask)
        
        for label_id in np.unique(label_mask):
            if label_id == 0: continue
            single_cell_binary = (label_mask == label_id).astype(np.uint8)
            eroded_single = cv2.erode(single_cell_binary, kernel, iterations=1)
            eroded_mask[eroded_single > 0] = label_id
            
        eroded_label_mask = measure.label(eroded_mask).astype(np.int32)
        return compute_mask_stats(eroded_label_mask)
    except Exception as e:
        print(f"Error aplicando erosión desde disco a {filename}: {e}")
        return None


# -------------------------------------------------------------------------------------
# FUNCIONES DE GRAFICADO ESTADÍSTICO
# -------------------------------------------------------------------------------------

def plot_temporal_metrics_with_ks(stats_dir, json_files=None, save_dir=None, save_name=None, 
                                  min_ecc=None, max_ecc=None, min_area=None, max_area=None, 
                                  label_mapping=None, smooth_lines=True, smoothing_sigma=2,
                                  ks_eval_points=None, aggregate_by_frame=False,
                                  group_mapping=None, pixel_size_um=None, 
                                  baseline_json_files=None, baseline_calc_pts=5, norm_mode=None,
                                  erosion_radius_plots=0): 
    # norm_mode puede ser: 'percent', 'absolute', o None
    
    all_data = []
    raw_distributions = {}

    if json_files is None:
        json_files = [f for f in os.listdir(stats_dir) if f.endswith(".json")]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    metrics_list = ["mean_area", "mean_solidity", "mean_eccentricity", "mean_orientation", "mean_junction_fraction"]
    loaded_experiments = [] 
    min_len = float('inf')

    print("--- Analyzing experiment lengths ---")
    for jf in json_files:
        path = os.path.join(stats_dir, jf)
        with open(path, "r") as f:
            data = json.load(f)
            folder_name = data["folder"]
            tp_sorted = natural_sort(list(data["timepoints"].keys()))
            
            if label_mapping is not None and jf in label_mapping:
                display_label = label_mapping[jf]
            else:
                display_label = folder_name
            
            current_len = len(tp_sorted)
            if current_len < min_len:
                min_len = current_len
            
            print(f"File: {display_label} -> Points: {current_len}")
            loaded_experiments.append((data, tp_sorted, display_label))
    
    print(f"==> CROPPING ALL EXPERIMENTS TO {min_len} TIMEPOINTS <==\n")

    for idx_f, (data, tp_sorted, exp_label) in enumerate(loaded_experiments):
        interval = data["interval"] 
        timepoints = data["timepoints"]
        main_jf = json_files[idx_f]
        
        raw_distributions[exp_label] = {m: [] for m in metrics_list}
        tp_to_process = tp_sorted[:min_len]

        # --- EXTRACCIÓN DE LA LÍNEA BASE (BASELINE) ---
        base_scalars = {m: None for m in metrics_list}
        if norm_mode in ['percent', 'absolute']:
            b_file = baseline_json_files[idx_f] if baseline_json_files else main_jf
            b_path = os.path.join(stats_dir, b_file)
            
            if os.path.exists(b_path):
                with open(b_path, "r") as f:
                    b_data = json.load(f)
                b_tps = natural_sort(list(b_data["timepoints"].keys()))
                
                if baseline_json_files:
                    b_eval_tps = b_tps[-baseline_calc_pts:] if baseline_calc_pts else b_tps
                else:
                    b_eval_tps = b_tps[:baseline_calc_pts] if baseline_calc_pts else b_tps
                    
                b_accum = {m: [] for m in metrics_list}
                area_mult = (pixel_size_um ** 2) if pixel_size_um is not None else 1.0
                
                for bt in b_eval_tps:
                    bt_raw = b_data["timepoints"][bt]["all_values"]
                    
                    # --- APLICACIÓN DE EROSIÓN REAL EN BASELINE ---
                    if erosion_radius_plots > 0:
                        eroded_stats = get_eroded_stats_from_disk(b_data["folder"], bt, erosion_radius_plots)
                        if eroded_stats is not None:
                            bt_raw = {
                                "mean_area": [s["area"] for s in eroded_stats],
                                "mean_solidity": [s["solidity"] for s in eroded_stats],
                                "mean_eccentricity": [s["eccentricity"] for s in eroded_stats],
                                "mean_orientation": [s["orientation"] for s in eroded_stats],
                                "mean_junction_fraction": [s["junction_fraction"] for s in eroded_stats]
                            }

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
                    if b_accum[m]:
                        base_scalars[m] = np.mean(b_accum[m])
        # -----------------------------------------------

        exp_time_data = []

        for idx, t in enumerate(tp_to_process):
            tp_data = timepoints[t]
            raw_vals = tp_data["all_values"]
            
            # --- APLICACIÓN DE EROSIÓN REAL EN PLOTS ---
            if erosion_radius_plots > 0:
                eroded_stats = get_eroded_stats_from_disk(data["folder"], t, erosion_radius_plots)
                if eroded_stats is not None:
                    raw_vals = {
                        "mean_area": [s["area"] for s in eroded_stats],
                        "mean_solidity": [s["solidity"] for s in eroded_stats],
                        "mean_eccentricity": [s["eccentricity"] for s in eroded_stats],
                        "mean_orientation": [s["orientation"] for s in eroded_stats],
                        "mean_junction_fraction": [s["junction_fraction"] for s in eroded_stats]
                    }
            
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
                    if i < len(r_junc):
                        filt_junc.append(r_junc[i])
                    
            exp_time_data.append({
                "idx": idx,
                "filename": t, 
                "filt_area": filt_area,
                "filt_sol": filt_sol,
                "filt_ecc": filt_ecc,
                "filt_ori": filt_ori,
                "filt_junc": filt_junc 
            })

        try:
            first_time_val = float(tp_sorted[0].split('_')[0])
        except:
            first_time_val = 0.0

        for tp in exp_time_data:
            idx = tp["idx"]
            t_filename = tp["filename"]
            
            try:
                time_val = float(t_filename.split('_')[0])
                if time_val > 1000000:
                    real_time_hours = (time_val - first_time_val) / 3600000.0
                else:
                    real_time_hours = time_val
            except:
                real_time_hours = idx * interval / 60.0

            raw_m_area = np.mean(tp["filt_area"]) if tp["filt_area"] else np.nan
            raw_m_sol = np.mean(tp["filt_sol"]) if tp["filt_sol"] else np.nan
            raw_m_ecc = np.mean(tp["filt_ecc"]) if tp["filt_ecc"] else np.nan
            raw_m_ori = np.mean(tp["filt_ori"]) if tp["filt_ori"] else np.nan
            raw_m_junc = np.mean(tp["filt_junc"]) if tp["filt_junc"] else np.nan

            # --- APLICAR NORMALIZACIÓN ---
            normalized_means = {}
            for m, raw_val in zip(metrics_list, [raw_m_area, raw_m_sol, raw_m_ecc, raw_m_ori, raw_m_junc]):
                b_val = base_scalars[m]
                if pd.isna(raw_val):
                    normalized_means[m] = np.nan
                elif norm_mode == 'percent' and b_val is not None and b_val != 0:
                    normalized_means[m] = ((raw_val - b_val) / b_val) * 100.0
                elif norm_mode == 'absolute' and b_val is not None:
                    normalized_means[m] = raw_val - b_val
                else:
                    normalized_means[m] = raw_val

            new_mean_area = normalized_means["mean_area"]
            new_mean_sol = normalized_means["mean_solidity"]
            new_mean_ecc = normalized_means["mean_eccentricity"]
            new_mean_ori = normalized_means["mean_orientation"]
            new_mean_junc = normalized_means["mean_junction_fraction"]
            
            all_data.append({
                "experiment": exp_label,
                "timepoint_idx": idx,
                "real_time_hours": real_time_hours, 
                "mean_area": new_mean_area,
                "mean_solidity": new_mean_sol,
                "mean_eccentricity": new_mean_ecc,
                "mean_orientation": new_mean_ori,
                "mean_junction_fraction": new_mean_junc 
            })

            if ks_eval_points is None or idx >= (min_len - ks_eval_points):
                if aggregate_by_frame:
                    if not np.isnan(new_mean_area): raw_distributions[exp_label]["mean_area"].append(new_mean_area)
                    if not np.isnan(new_mean_sol): raw_distributions[exp_label]["mean_solidity"].append(new_mean_sol)
                    if not np.isnan(new_mean_ecc): raw_distributions[exp_label]["mean_eccentricity"].append(new_mean_ecc)
                    if not np.isnan(new_mean_ori): raw_distributions[exp_label]["mean_orientation"].append(new_mean_ori)
                    if not np.isnan(new_mean_junc): raw_distributions[exp_label]["mean_junction_fraction"].append(new_mean_junc)
                else:
                    raw_distributions[exp_label]["mean_area"].append(new_mean_area)
                    raw_distributions[exp_label]["mean_solidity"].append(new_mean_sol)
                    raw_distributions[exp_label]["mean_eccentricity"].append(new_mean_ecc)
                    raw_distributions[exp_label]["mean_orientation"].append(new_mean_ori)
                    raw_distributions[exp_label]["mean_junction_fraction"].append(new_mean_junc)

    df = pd.DataFrame(all_data)
    df = df.sort_values(["experiment", "real_time_hours"]) 
    experiment_names = list(raw_distributions.keys())

    filter_suffix = ""
    if min_ecc is not None: filter_suffix += f"_minEcc{min_ecc}"
    if max_ecc is not None: filter_suffix += f"_maxEcc{max_ecc}"
    if min_area is not None: filter_suffix += f"_minArea{min_area}"
    if max_area is not None: filter_suffix += f"_maxArea{max_area}"
    if norm_mode == 'percent': filter_suffix += "_pct"
    if norm_mode == 'absolute': filter_suffix += "_abs"
    if erosion_radius_plots > 0: filter_suffix += f"_erode{erosion_radius_plots}px"

    for metric in metrics_list:
        plt.figure(figsize=(12, 8)) 
        
        for exp in experiment_names:
            df_exp = df[df["experiment"] == exp].sort_values("real_time_hours")
            
            x_vals = df_exp["real_time_hours"].values 
            y_vals = df_exp[metric].values

            if smooth_lines:
                mask = ~np.isnan(y_vals)
                x_clean = x_vals[mask]
                y_clean = y_vals[mask]

                if len(x_clean) > 1:
                    try:
                        y_smooth = gaussian_filter1d(y_clean, sigma=smoothing_sigma)
                        p = plt.plot(x_clean, y_smooth, label=exp, linewidth=2)
                        plt.plot(x_clean, y_clean, "o", color=p[0].get_color(), alpha=0.3, markersize=4)
                    except Exception as e:
                        print(f"Warning: Could not smooth {exp}. Error: {e}")
                        plt.plot(x_vals, y_vals, marker="o", label=exp)
                else:
                    plt.plot(x_vals, y_vals, marker="o", label=exp)
            else:
                plt.plot(x_vals, y_vals, marker="o", label=exp)

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        
        # --- ETIQUETAS DINÁMICAS SEGÚN EL MODO DE NORMALIZACIÓN ---
        base_metric_name = metric.replace("mean_", "").replace("_", " ").title()
        if norm_mode == 'percent':
            ylabel_text = f"Change in {base_metric_name} (%)"
        elif norm_mode == 'absolute':
            ylabel_text = f"Δ {base_metric_name} (Absolute)"
        else:
            if metric == "mean_area":
                ylabel_text = "Mean Area (μm²)" if pixel_size_um is not None else "Mean Area (pixels)"
            elif metric == "mean_junction_fraction":
                ylabel_text = "Junction Fraction (0 to 1)"
            else:
                ylabel_text = base_metric_name

        plt.ylabel(ylabel_text)
        plt.xlabel("Time (hours)") 
        
        eval_text = f" (Eval on last {ks_eval_points} pts)" if ks_eval_points is not None else ""
        agg_text = " [Means]" if aggregate_by_frame else " [All Cells]"
        grp_text = " [Grouped Test]" if group_mapping is not None else ""
        norm_text = f" [Baseline: {baseline_calc_pts} pts]" if norm_mode else ""
        erosion_text = f" [Erosion: {erosion_radius_plots}px]" if erosion_radius_plots > 0 else ""
        
        plt.title(f"{base_metric_name} Evolution (Cropped to {min_len} points){filter_suffix.replace('_', ' ')}{eval_text}{agg_text}{grp_text}{norm_text}{erosion_text}")

        if group_mapping is not None:
            distributions_to_test = {}
            for exp in experiment_names:
                grp = group_mapping.get(exp, exp)
                if grp not in distributions_to_test:
                    distributions_to_test[grp] = {m: [] for m in metrics_list}
                distributions_to_test[grp][metric].extend(raw_distributions[exp][metric])
            test_names = list(distributions_to_test.keys())
        else:
            distributions_to_test = raw_distributions
            test_names = experiment_names

        stat_results_text = []
        
        if len(experiment_names) >= 2:
            print(f"\n--- Mann-Whitney U Test Results (Individual) for {metric} ---")
            for exp1, exp2 in combinations(experiment_names, 2):
                vals1 = raw_distributions[exp1][metric]
                vals2 = raw_distributions[exp2][metric]
                
                if len(vals1) > 0 and len(vals2) > 0:
                    stat, pval = mannwhitneyu(vals1, vals2, alternative='two-sided')
                    print(f"{exp1} vs {exp2} -> stat(U)={stat:.3f}, p={pval:.3e}")
                    
                    short_exp1 = exp1[:15] + ".." if len(exp1) > 15 else exp1
                    short_exp2 = exp2[:15] + ".." if len(exp2) > 15 else exp2
                    
                    if pval < 0.05:
                        if pval < 0.001: sig_level = "***"
                        elif pval < 0.01: sig_level = "**"
                        else: sig_level = "*"
                        
                        raw_str = f"{short_exp1} vs {short_exp2}: {sig_level} (p={pval:.3f})"
                        safe_str = raw_str.replace(' ', r'\ ').replace('_', r'\_')
                        final_str = r"$\bf{" + safe_str + "}$"
                    else:
                        final_str = f"{short_exp1} vs {short_exp2}: ns (p={pval:.3f})"
                        
                    stat_results_text.append(final_str)

        if group_mapping is not None:
            distributions_to_test = {}
            for exp in experiment_names:
                grp = group_mapping.get(exp, exp)
                if grp not in distributions_to_test:
                    distributions_to_test[grp] = {m: [] for m in metrics_list}
                distributions_to_test[grp][metric].extend(raw_distributions[exp][metric])
            
            test_names_grouped = list(distributions_to_test.keys())
            
            if len(test_names_grouped) >= 2:
                print(f"\n--- Mann-Whitney U Test Results (Grouped) for {metric} ---")
                stat_results_text.append("--- Grouped ---") 
                
                for grp1, grp2 in combinations(test_names_grouped, 2):
                    vals1 = distributions_to_test[grp1][metric]
                    vals2 = distributions_to_test[grp2][metric]
                    
                    if len(vals1) > 0 and len(vals2) > 0:
                        stat, pval = mannwhitneyu(vals1, vals2, alternative='two-sided')
                        print(f"{grp1} vs {grp2} -> stat(U)={stat:.3f}, p={pval:.3e}")
                        
                        short_grp1 = grp1[:15] + ".." if len(grp1) > 15 else grp1
                        short_grp2 = grp2[:15] + ".." if len(grp2) > 15 else grp2
                        
                        if pval < 0.05:
                            if pval < 0.001: sig_level = "***"
                            elif pval < 0.01: sig_level = "**"
                            else: sig_level = "*"
                            
                            raw_str = f"{short_grp1} vs {short_grp2}: {sig_level} (p={pval:.3f})"
                            safe_str = raw_str.replace(' ', r'\ ').replace('_', r'\_')
                            final_str = r"$\bf{" + safe_str + "}$"
                        else:
                            final_str = f"{short_grp1} vs {short_grp2}: ns (p={pval:.3f})"
                            
                        stat_results_text.append(final_str)
        
        if stat_results_text:
            display_text = "All Comparisons\n(Bold = Significant p<0.05):\n" + "\n".join(stat_results_text)
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax.text(1.02, 0.98, display_text, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', bbox=props)
            plt.subplots_adjust(right=0.65) 

        if norm_mode in ['percent', 'absolute']:
            plt.axhline(0, color='gray', linestyle='--', alpha=0.7)

        plt.legend(title="Condition", loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                    fancybox=True, shadow=False, ncol=3)
        
        plt.tight_layout(rect=[0, 0, 0.75, 1]) 

        if save_dir is not None:
            metric_folder = os.path.join(save_dir, metric)
            os.makedirs(metric_folder, exist_ok=True)
            filename = f"{save_name}_{metric}{filter_suffix}.jpg"
            save_path = os.path.join(metric_folder, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")

        plt.show()

def plot_polar_orientation(stats_dir, json_files=None, save_dir=None, save_name="angles", min_ecc=None, max_ecc=None, 
                           label_mapping=None, ks_eval_points=None, aggregate_by_frame=False, group_mapping=None,
                           erosion_radius_plots=0):
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
            
            if label_mapping is not None and jf in label_mapping:
                display_label = label_mapping[jf]
            else:
                display_label = folder_name
                
            if len(tp_sorted) < min_len:
                min_len = len(tp_sorted)
                
            loaded_experiments.append((data, tp_sorted, display_label))

    raw_angle_distributions = {}

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1) 

    palette = sns.color_palette("bright", len(json_files))

    for idx, (data, tp_sorted, display_label) in enumerate(loaded_experiments):
        all_angles_for_plot = []
        frame_means_for_test = []
        folder_name = data["folder"]
        
        timepoints = data["timepoints"]
        
        if ks_eval_points is not None:
            start_idx = max(0, min_len - ks_eval_points)
            tp_to_process = tp_sorted[start_idx:min_len]
        else:
            tp_to_process = tp_sorted[:min_len]

        try:
            for t in tp_to_process:
                raw_vals = timepoints[t]["all_values"]
                
                # --- APLICACIÓN DE EROSIÓN REAL EN POLAR ---
                if erosion_radius_plots > 0:
                    eroded_stats = get_eroded_stats_from_disk(data["folder"], t, erosion_radius_plots)
                    if eroded_stats is not None:
                        raw_vals = {
                            "mean_orientation": [s["orientation"] for s in eroded_stats],
                            "mean_eccentricity": [s["eccentricity"] for s in eroded_stats]
                        }

                angles = raw_vals["mean_orientation"]
                eccentricities = raw_vals["mean_eccentricity"]
                
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
                print(f"Warning: No cells left in {display_label}")
                raw_angle_distributions[display_label] = []
                continue

            angles_deg_plot = np.array(all_angles_for_plot)
            
            if aggregate_by_frame:
                raw_angle_distributions[display_label] = np.deg2rad(np.array(frame_means_for_test))
            else:
                raw_angle_distributions[display_label] = np.deg2rad(angles_deg_plot)

            angles_rad_plot = np.deg2rad(angles_deg_plot)
            angles_rad_symmetric = np.concatenate([angles_rad_plot, angles_rad_plot + np.pi])
            angles_rad_symmetric = angles_rad_symmetric % (2 * np.pi)

            bins_number = 36 
            counts, bin_edges = np.histogram(angles_rad_symmetric, bins=bins_number, range=(0, 2*np.pi), density=True)
            
            theta = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts = np.append(counts, counts[0])
            theta = np.append(theta, theta[0])

            ax.plot(theta, counts, color=palette[idx], linewidth=2, label=display_label)
            ax.fill(theta, counts, color=palette[idx], alpha=0.1)

        except Exception as e:
            print(f"Error processing: {e}")

    filter_info = ""
    if min_ecc is not None: filter_info += f" Min Ecc: {min_ecc}"
    if max_ecc is not None: filter_info += f" Max Ecc: {max_ecc}"
    
    eval_text = f"\n(Eval on last {ks_eval_points} pts)" if ks_eval_points is not None else ""
    agg_text = " [Means]" if aggregate_by_frame else " [All Cells]"
    erosion_text = f" Erosion: {erosion_radius_plots}px" if erosion_radius_plots > 0 else ""
    plt.title(f"Orientation Distribution (Cropped to {min_len} tps)\n{save_name}{filter_info}{erosion_text}{eval_text}{agg_text}", y=1.08)
    
    kuiper_results_text = []
    experiment_names = list(raw_angle_distributions.keys())
    
    if len(experiment_names) >= 2:
        print(f"\n--- Kuiper Test Results (Individual) for Polar Orientation ({save_name}) ---")
        for exp1, exp2 in combinations(experiment_names, 2):
            vals1 = raw_angle_distributions[exp1]
            vals2 = raw_angle_distributions[exp2]
            
            if len(vals1) > 0 and len(vals2) > 0:
                stat, pval = kuiper_two(vals1, vals2)
                print(f"{exp1} vs {exp2} -> stat(V)={stat:.3f}, p={pval:.3e}")
                
                short_exp1 = exp1[:15] + ".." if len(exp1) > 15 else exp1
                short_exp2 = exp2[:15] + ".." if len(exp2) > 15 else exp2
                
                if pval < 0.05:
                    if pval < 0.001: sig_level = "***"
                    elif pval < 0.01: sig_level = "**"
                    else: sig_level = "*"
                    
                    raw_str = f"{short_exp1} vs {short_exp2}: {sig_level} (p={pval:.3f})"
                    safe_str = raw_str.replace(' ', r'\ ').replace('_', r'\_')
                    final_str = r"$\bf{" + safe_str + "}$"
                else:
                    final_str = f"{short_exp1} vs {short_exp2}: ns (p={pval:.3f})"
                    
                kuiper_results_text.append(final_str)

    if group_mapping is not None:
        distributions_to_test = {}
        for exp in experiment_names:
            grp = group_mapping.get(exp, exp)
            if grp not in distributions_to_test:
                distributions_to_test[grp] = []
            distributions_to_test[grp].extend(raw_angle_distributions[exp])
        
        test_names_grouped = list(distributions_to_test.keys())
        
        if len(test_names_grouped) >= 2:
            print(f"\n--- Kuiper Test Results (Grouped) for Polar Orientation ({save_name}) ---")
            kuiper_results_text.append("--- Grouped ---")
            
            for grp1, grp2 in combinations(test_names_grouped, 2):
                vals1 = distributions_to_test[grp1]
                vals2 = distributions_to_test[grp2]
                
                if len(vals1) > 0 and len(vals2) > 0:
                    stat, pval = kuiper_two(vals1, vals2)
                    print(f"{grp1} vs {grp2} -> stat(V)={stat:.3f}, p={pval:.3e}")
                    
                    short_grp1 = grp1[:15] + ".." if len(grp1) > 15 else grp1
                    short_grp2 = grp2[:15] + ".." if len(grp2) > 15 else grp2
                    
                    if pval < 0.05:
                        if pval < 0.001: sig_level = "***"
                        elif pval < 0.01: sig_level = "**"
                        else: sig_level = "*"
                        
                        raw_str = f"{short_grp1} vs {short_grp2}: {sig_level} (p={pval:.3f})"
                        safe_str = raw_str.replace(' ', r'\ ').replace('_', r'\_')
                        final_str = r"$\bf{" + safe_str + "}$"
                    else:
                        final_str = f"{short_grp1} vs {short_grp2}: ns (p={pval:.3f})"
                        
                    kuiper_results_text.append(final_str)

    if kuiper_results_text:
        display_text = "All Comparisons\n(Bold = Significant p<0.05):\n" + "\n".join(kuiper_results_text)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(1.20, 1.05, display_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=props)
        plt.subplots_adjust(right=0.65) 

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    lines, labels = plt.thetagrids(range(0, 360, 45), labels=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    plt.tight_layout(rect=[0, 0, 0.75, 1]) 

    if save_dir is not None:
        metric = "polar_lines"
        metric_folder = os.path.join(save_dir, metric)
        os.makedirs(metric_folder, exist_ok=True)
        suffix = ""
        if min_ecc is not None: suffix += f"_min{min_ecc}"
        if max_ecc is not None: suffix += f"_max{max_ecc}"
        if erosion_radius_plots > 0: suffix += f"_erode{erosion_radius_plots}px"
        filename = f"{save_name}_polar_lines{suffix}.jpg"
        save_path = os.path.join(metric_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved polar plot: {save_path}")

    plt.show()







# -------------------------------------------------------------------------------------
# EJECUCIÓN ESTADÍSTICA Y GRÁFICAS
# -------------------------------------------------------------------------------------

stats_dir = "./segmentation_stats"
save_dir = "./plots"
fApplySelection = 1

lastTimePointsToEval = 24 
lastTimePointsToEval = 1000
aggregateByFrame=True 

# --- CONTROL DE EROSIÓN REAL EN LOS PLOTS ---
# 0 = No erosiona (lee rápido del JSON).
# 1, 2, 3... = Carga la máscara comprimida .npz, le pasa cv2.erode circular y pinta los datos actualizados.
erosion_radius_plots = 1
# ---------------------------------------

fDo = 0
if fDo == 1:
    # --- Crecimiento ---
    json_files = ["CRU002_CRU002 - CH1 - 0.json", 
                  "CRU003_CRU003 - CH1 - 0.json", 
                  "CRU004_CRU004 - CH1 - 0.json"]      
    labels = ["HLT CRU002 GROWTH", 
              "BQL CRU003 GROWTH", 
              "BQL CRU004 GROWTH"]     
    labels = dict(zip(json_files, labels))            
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "crecimiento", label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, erosion_radius_plots=erosion_radius_plots)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="crecimiento", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, erosion_radius_plots=erosion_radius_plots)

fDo = 0
if fDo == 1:
    # --- Before ---
    json_files = ["CRU002_CRU002 - CH1 - 1 - BEFORE.json", 
                  "CRU003_CRU003 - CH2 - 1 - BEFORE.json", 
                  "CRU004_CRU004 - CH1 - 1 - BEFORE.json",
                  "CRU008_CRU008 - CH1 - 1 - BEFORE.json",
                  "CRU010_CRU010 - CH1 - 1 - BEFORE.json",
                  "CRU012_CRU012 - CH2 - 1 - BEFORE.json"]
    labels = ["HLT CRU002 BEFORE", 
              "BQL CRU003 BEFORE", 
              "BQL CRU004 BEFORE",
              "HLT CRU008 BEFORE",
              "HLT CRU010 BEFORE",
              "HLT CRU012 BEFORE"]
    if fApplySelection == 1:
        indices = (0, 1, 2, 3)
        json_files = [json_files[i] for i in indices]
        labels = [labels[i] for i in indices]
        
    labels = dict(zip(json_files, labels))                 
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "before", label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, erosion_radius_plots=erosion_radius_plots)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="before", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, erosion_radius_plots=erosion_radius_plots)

fDo = 0
if fDo == 1:
    # --- Trigger ---
    json_files = ["CRU002_CRU002 - CH1 - 2 - TRIGGER - HLT.json", 
                  "CRU003_CRU003 - CH2 - 2 - TRIGGER - BQL.json", 
                  "CRU004_CRU004 - CH1 - 2 - TRIGGER - BQL.json",
                  "CRU008_CRU008 - CH1 - 2 - TRIGGER - HLT.json",
                  "CRU010_CRU010 - CH1 - 2 - TRIGGER - HLT.json",
                  "CRU012_CRU012 - CH2 - 2 - TRIGGER - HLT.json"]
    labels = ["HLT CRU002 TRIGGER", 
              "BQL CRU003 TRIGGER", 
              "BQL CRU004 TRIGGER",
              "HLT CRU008 TRIGGER",
              "HLT CRU010 TRIGGER",
              "HLT CRU012 TRIGGER"]
    if fApplySelection == 1:
        indices = (0, 1, 2, 3)
        json_files = [json_files[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    labels = dict(zip(json_files, labels))                
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "trigger", label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, erosion_radius_plots=erosion_radius_plots)
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="trigger", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, erosion_radius_plots=erosion_radius_plots)


   
fDo = 1
if fDo == 1:
    # --- After ---
    json_files = ["CRU002_CRU002 - CH1 - 3 - AFTER.json", 
                  "CRU003_CRU003 - CH2 - 3 - AFTER.json", 
                  "CRU004_CRU004 - CH1 - 3 - AFTER.json",
                  "CRU008_CRU008 - CH1 - 3 - AFTER.json",
                  "CRU010_CRU010 - CH1 - 3 - AFTER.json",
                  "CRU012_CRU012 - CH2 - 3 - AFTER.json"]   
                  
    # Lista de archivos base correspondientes a la fase BEFORE
    baseline_files = ["CRU002_CRU002 - CH1 - 1 - BEFORE.json", 
                      "CRU003_CRU003 - CH2 - 1 - BEFORE.json", 
                      "CRU004_CRU004 - CH1 - 1 - BEFORE.json",
                      "CRU008_CRU008 - CH1 - 1 - BEFORE.json",
                      "CRU010_CRU010 - CH1 - 1 - BEFORE.json",
                      "CRU012_CRU012 - CH2 - 1 - BEFORE.json"]
    
    labels = ["HLT CRU002 AFTER", 
              "BQL CRU003 AFTER", 
              "BQL CRU004 AFTER",
              "HLT CRU008 AFTER",
              "HLT CRU010 AFTER",
              "HLT CRU012 AFTER"]  
              
    gm = {"HLT CRU002 AFTER": "HLT (Control)",
        "BQL CRU003 AFTER": "BQL (Infected)",
        "BQL CRU004 AFTER": "BQL (Infected)",
        "HLT CRU008 AFTER": "HLT (Control)",
        "HLT CRU010 AFTER": "HLT (Control)",
        "HLT CRU012 AFTER": "HLT (Control)"}
    
    if fApplySelection == 1:
        indices = (0, 1, 2, 3)
        json_files = [json_files[i] for i in indices]
        baseline_files = [baseline_files[i] for i in indices] # Sincronizamos los baselines
        labels = [labels[i] for i in indices]
        gm = {"HLT CRU002 AFTER": "HLT (Control)",
            "BQL CRU003 AFTER": "BQL (Infected)",
            "BQL CRU004 AFTER": "BQL (Infected)",
            "HLT CRU008 AFTER": "HLT (Control)"}

    labels = dict(zip(json_files, labels))   
    
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "after_vs_before", 
                                  max_area=450, label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm,
                                  baseline_json_files=baseline_files, baseline_calc_pts=15, norm_mode='percent', erosion_radius_plots=erosion_radius_plots)
    
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "after_vs_before", 
                                  min_ecc=0.85, max_area=450, label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm,
                                  baseline_json_files=baseline_files, baseline_calc_pts=15, norm_mode='percent', erosion_radius_plots=erosion_radius_plots)
    
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name = "after_vs_before", 
                                  max_ecc=0.85, max_area=450, label_mapping=labels, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm,
                                  baseline_json_files=baseline_files, baseline_calc_pts=15, norm_mode='percent', erosion_radius_plots=erosion_radius_plots)
    
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="after", label_mapping=labels, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, group_mapping=gm, erosion_radius_plots=erosion_radius_plots)
    
    
fDo = 1
if fDo == 1:    

    # --- Validación Crecimiento 96 Horas (CRU000) ---
    json_files = ["CRU000_timepoints_0_4_24_48_72_96_CRU000.json"]
    
    # Etiqueta que aparecerá en la leyenda de las gráficas
    labels = ["CRU000 Growth (96h)"] 
    
    # Creamos el diccionario de mapeo
    labels_dict = dict(zip(json_files, labels))
    
    # 1. Gráfica estándar (sin filtros)
    plot_temporal_metrics_with_ks(stats_dir, json_files=json_files, save_dir=save_dir, save_name="cru000_96h", 
                                  label_mapping=labels_dict, 
                                  ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, erosion_radius_plots=erosion_radius_plots)
    
    # 2. Gráfica polar de orientación
    plot_polar_orientation(stats_dir, json_files=json_files, save_dir=save_dir, save_name="cru000_96h", 
                           label_mapping=labels_dict, 
                           ks_eval_points=lastTimePointsToEval, aggregate_by_frame=aggregateByFrame, erosion_radius_plots=erosion_radius_plots)

print("\nProcesamiento estadístico completado.")