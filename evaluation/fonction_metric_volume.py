"""
Created on Wed Feb 26 10:34:20 2025

@author: Miguel Castro USER
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

def calculate_volumes(segmentation_pred, ground_truth, labels, voxel_sizes):
    # Calculer le volume d'un voxel en mm³
    voxel_volume_mm3 = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
    
    # Convertir le volume d'un voxel en cc (1 cc = 1000 mm³)
    voxel_volume_cc = voxel_volume_mm3 / 1000.0
    
    # Initialisation des volumes pour chaque étiquette/label (sauf background, donc -1)
    volumes = np.zeros(labels - 1)
    vol_ref = np.zeros(labels - 1)
    
    for etiquette in range(1, labels):  # Commencer à 1 car pas besoin du background qui est le 0
        if etiquette == 6:
            bladderneck = (segmentation_pred == 6) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 6) | (ground_truth == 10)
            # volumes[etiquette - 1] = np.abs(np.sum(bladderneck) -np.sum(verite_terrain))* voxel_volume_cc
            volumes[etiquette - 1] = np.sum(bladderneck) * voxel_volume_cc
            vol_ref[etiquette - 1] = np.sum(verite_terrain) * voxel_volume_cc
        elif etiquette == 7:
            bladdertrigone = (segmentation_pred == 7) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 7) | (ground_truth == 10)
            # volumes[etiquette - 1] = np.abs(np.sum(bladdertrigone) -np.sum(verite_terrain))* voxel_volume_cc
            volumes[etiquette - 1] = np.sum(bladdertrigone) * voxel_volume_cc
            vol_ref[etiquette - 1] = np.sum(verite_terrain) * voxel_volume_cc
        else:
            # volumes[etiquette - 1] = np.abs(np.sum(segmentation_pred == etiquette) -np.sum(ground_truth == etiquette) )* voxel_volume_cc
            volumes[etiquette - 1] = np.sum(segmentation_pred == etiquette) * voxel_volume_cc
            vol_ref[etiquette - 1] = np.sum(ground_truth == etiquette)* voxel_volume_cc
    print(volumes)
    return volumes, vol_ref
    
# Mean Surface Distance (MSD)     
def mean_surface_distance(segmentation_pred, ground_truth, labels,  voxel_spacing):
   
    # Initialisation de Mean Surface Distance (MSD) pour chaque étiquette/label (sauf background, donc -1)
    msd = np.zeros(labels - 1)
   
    for etiquette in range(1, labels):  # Commencer à 1 car pas besoin du background qui est le 0
        if etiquette == 6:
            bladderneck = (segmentation_pred == 6) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 6) | (ground_truth == 10)
  
            pred_surface = find_boundaries(bladderneck, mode='inner')
            gt_surface = find_boundaries(verite_terrain, mode='inner')
          
            # Distance map of ground truth surface
            dt_gt = distance_transform_edt(~gt_surface, sampling=voxel_spacing)
            dt_pred = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
            
            surface_dist_pred_to_gt = dt_gt[pred_surface]
            surface_dist_gt_to_pred = dt_pred[gt_surface]
            
            msd[etiquette - 1] = (np.mean(surface_dist_pred_to_gt) + np.mean(surface_dist_gt_to_pred)) / 2.0 
 
            
        elif etiquette == 7:
            bladdertrigone = (segmentation_pred == 7) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 7) | (ground_truth == 10)
            
            pred_surface = find_boundaries(bladdertrigone, mode='inner')
            gt_surface = find_boundaries(verite_terrain, mode='inner')
  
            # Distance map of ground truth surface
            dt_gt = distance_transform_edt(~gt_surface, sampling=voxel_spacing)
            dt_pred = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
            
            surface_dist_pred_to_gt = dt_gt[pred_surface]
            surface_dist_gt_to_pred = dt_pred[gt_surface]
            
            msd[etiquette - 1] = (np.mean(surface_dist_pred_to_gt) + np.mean(surface_dist_gt_to_pred)) / 2.0 
        else:

            struct = (segmentation_pred == etiquette)
            verite_terrain = (ground_truth == etiquette)
            
            pred_surface = find_boundaries(struct, mode='inner')
            gt_surface = find_boundaries(verite_terrain, mode='inner')
  
            # Distance map of ground truth surface
            dt_gt = distance_transform_edt(~gt_surface, sampling=voxel_spacing)
            dt_pred = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
            
            surface_dist_pred_to_gt = dt_gt[pred_surface]
            surface_dist_gt_to_pred = dt_pred[gt_surface]
            
            msd[etiquette - 1] = (np.mean(surface_dist_pred_to_gt) + np.mean(surface_dist_gt_to_pred)) / 2.0 
    print(msd)

    
    return msd
    
# Relative Volume Difference (RVD)     
def relative_volume_difference(segmentation_pred, ground_truth, labels):
   
    # Initialisation de Relative Volume Difference (RVD) pour chaque étiquette/label (sauf background, donc -1)
    rvd = np.zeros(labels - 1)
   
    for etiquette in range(1, labels):  # Commencer à 1 car pas besoin du background qui est le 0
        if etiquette == 6:
            bladderneck = (segmentation_pred == 6) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 6) | (ground_truth == 10)
            volume = np.sum(bladderneck)
            vol_ref = np.sum(verite_terrain)
            rvd[etiquette - 1] = (volume - vol_ref)/vol_ref
            
        elif etiquette == 7:
            bladdertrigone = (segmentation_pred == 7) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 7) | (ground_truth == 10)
            
            volume = np.sum(bladdertrigone)
            vol_ref = np.sum(verite_terrain)
            rvd[etiquette - 1] = (volume - vol_ref)/vol_ref
        else:
            struct = (segmentation_pred == etiquette)
            verite_terrain = (ground_truth == etiquette)
            
            volume = np.sum(struct)
            vol_ref = np.sum(verite_terrain)
            rvd[etiquette - 1] = (volume - vol_ref)/vol_ref
    
    return rvd

# Volumetric Overlap Error (VOE)     
def volumetric_overlap_error(segmentation_pred, ground_truth, labels):
   
    # Initialisation de Volumetric Overlap Error (VOE) pour chaque étiquette/label (sauf background, donc -1)
    voe = np.zeros(labels - 1)
   
    for etiquette in range(1, labels):  # Commencer à 1 car pas besoin du background qui est le 0
        if etiquette == 6:
            bladderneck = (segmentation_pred == 6) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 6) | (ground_truth == 10)
            
            intersection = np.logical_and(bladderneck, verite_terrain).sum()
            union = np.logical_or(bladderneck, verite_terrain).sum()
            if union == 0:
                voe[etiquette - 1] = 0.0
            else:
                voe[etiquette - 1] = 1.0 - intersection / union
            
            
        elif etiquette == 7:
            bladdertrigone = (segmentation_pred == 7) | (segmentation_pred == 10)
            verite_terrain = (ground_truth == 7) | (ground_truth == 10)
            
            intersection = np.logical_and(bladdertrigone, verite_terrain).sum()
            union = np.logical_or(bladdertrigone, verite_terrain).sum()
            if union == 0:
                voe[etiquette - 1] = 0.0
            else:
                voe[etiquette - 1] = 1.0 - intersection / union
        else:
            struct = (segmentation_pred == etiquette)
            verite_terrain = (ground_truth == etiquette)
                      
            intersection = np.logical_and(struct, verite_terrain).sum()
            union = np.logical_or(struct, verite_terrain).sum()
            if union == 0:
                voe[etiquette - 1] = 0.0
            else:
                voe[etiquette - 1] = 1.0 - intersection / union
    
    return voe

