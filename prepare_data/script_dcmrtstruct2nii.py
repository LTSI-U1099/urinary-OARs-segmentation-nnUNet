# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:18:16 2025

@author: Miguel Castro USER
"""

#dcmrtstruct2nii('D:\\temps\\Patient1\\RS1.2.752.243.1.1.20241212135033889.3790.10673.dcm', 'D:\\temps\\Patient1', 'D:\\temps\\CEM1')

import os
import subprocess
from dcmrtstruct2nii import dcmrtstruct2nii

def find_largest_filename(directory):
    largest_file = None
    largest_size = 1000000000 # 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            if file_size < largest_size: # file_size > largest_size
                largest_size = file_size
                largest_file = file_path
    
    return largest_file

def process_subdirectories(base_directory, output_directory):
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            shortest_file = find_largest_filename(subdir_path)
            output_subdir = os.path.join(output_directory, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            
            print(subdir_path)
            dcmrtstruct2nii(shortest_file, subdir_path, output_subdir)
            

# Example usage
base_directory = 'D:\\CEM_0125'
output_directory = 'D:\\CEM'
process_subdirectories(base_directory, output_directory)