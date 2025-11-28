# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:11:23 2025

@author: Miguel Castro USER
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass, binary_fill_holes
from skimage.measure import label, regionprops
from scipy.interpolate import splprep, splev

# Draw a circle on the image with the given center and radius
def draw_circle(image, center, radius, label_value):
    rr, cc = np.ogrid[:image.shape[0], :image.shape[1]]
    circle = (rr - center[0])**2 + (cc - center[1])**2 <= radius**2
    image[circle] = label_value
    return image

# Function to create centerline representation for the specified labels
def create_centerline_representation(input_folder, output_folder, labels, new_label=3, extrapolated_label=3):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            # Load the NIfTI file
            filepath = os.path.join(input_folder, filename)
            img = nib.load(filepath)
            data = img.get_fdata()
            
            # Initialize the output mask
            centerline_intraprostaticurethra = np.zeros_like(data)
            radii = []
            points = []

            for target_label in labels:
                # Create binary mask from segmented image for the target label
                mask = (data == target_label).astype(np.uint8)
                
                # Fill holes in the mask to ensure continuity
                mask = binary_fill_holes(mask).astype(np.uint8)
                
                for z in (range(mask.shape[2])):  # Inverser l'ordre de la boucle
                    slice_mask = mask[:, :, z]
                    
                    if np.any(slice_mask):  # Check if the slice contains any foreground pixels
                        com_intra = center_of_mass(slice_mask)
                        centerline_intraprostaticurethra[int(com_intra[0]), int(com_intra[1]), z] = new_label   
                        points.append([com_intra[0], com_intra[1], z])
                        
                        # Label the regions in the slice
                        labeled_slice = label(slice_mask)
                        
                        # Calculate region properties
                        regions = regionprops(labeled_slice)
                        
                        for region in regions:
                            equivalent_diameter = region.equivalent_diameter
                            radius = equivalent_diameter / 2
                            radii.append(radius)
                           
                            
                            # Draw the circle around the center of mass
                            centerline_intraprostaticurethra[:, :, z] = draw_circle(centerline_intraprostaticurethra[:, :, z], com_intra, radius, new_label)
            
            # Extrapolate the curve
            points = np.array(points)
            tck, u = splprep(points.T, s=20, k=3)
            u_fine = np.linspace(0, 1, num=mask.shape[2])
            x_fine, y_fine, z_fine = splev(u_fine, tck)
            
            # Calculate the average radius
            average_radius = np.mean(radii)
            
            # Create the new label 14 mask
            extrapolated_mask = np.zeros_like(data)
            for x, y, z in zip(x_fine, y_fine, z_fine):
                # Find the closest original radius
                closest_z = int(np.round(z))
                if closest_z < len(radii):
                    original_radius = radii[closest_z]
                    if original_radius > average_radius:
                        radius_to_use = original_radius
                    else:
                        radius_to_use = average_radius
                else:
                    radius_to_use = average_radius
                
                extrapolated_mask[:, :, int(z)] = draw_circle(extrapolated_mask[:, :, int(z)], [int(x), int(y)], radius_to_use, extrapolated_label)
            
            # Continue extrapolation with average_radius until the end of the volume
            for z in range(int(z_fine[-1]), mask.shape[2]):
                extrapolated_mask[:, :, z] = draw_circle(extrapolated_mask[:, :, z], [int(x_fine[-1]), int(y_fine[-1])], average_radius, extrapolated_label)
            
            # Combine the extrapolated mask with the original data
            combined_mask = np.maximum(data, extrapolated_mask)
            
            # Save the centerline representations as new NIfTI files
            centerline_img = nib.Nifti1Image(centerline_intraprostaticurethra, img.affine)
            extrapolated_img = nib.Nifti1Image(extrapolated_mask, img.affine)
            combined_img = nib.Nifti1Image(combined_mask, img.affine)
            output_path_centerline = os.path.join(output_folder, f"{filename}_centerline_seg_label_{new_label}.nii.gz")
            output_path_extrapolated = os.path.join(output_folder, f"{filename}_extrapolated_seg_label_{extrapolated_label}.nii.gz")
            output_path_combined = os.path.join(output_folder, f"{filename}_combined_seg.nii.gz")
            nib.save(centerline_img, output_path_centerline)
            nib.save(extrapolated_img, output_path_extrapolated)
            nib.save(combined_img, output_path_combined)

# Define input and output folders
input_folder = 'D:\\TEMP\\input'
output_folder = 'D:\\TEMP\\output'

# Create centerline representations for labels 8, 7, and 3 and save as label 13, then extrapolate and save as label 14
create_centerline_representation(input_folder, output_folder, labels=[3], new_label=3, extrapolated_label=3)

print("Centerline, extrapolated, and combined representations created and saved successfully.")
