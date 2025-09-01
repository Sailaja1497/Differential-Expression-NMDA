"""

Author: Sailaja Kuruvada
Date: 2025


Fluorescence Intensity Analysis Tool

This script analyzes fluorescence intensity in microscopy images by performing background subtraction,
intensity measurements, and batch processing of multiple TIFF files. It extracts key metrics including
mean intensity, integrated density, and area measurements for quantitative analysis of fluorescent
protein expression patterns in various biological samples.

Key Features:
- Background subtraction using rolling ball algorithm
- Gaussian smoothing for noise reduction
- Otsu thresholding for automatic segmentation
- Intensity and area measurements
- Batch processing of multiple TIFF files
- Metadata parsing from filenames
- Comprehensive results export to CSV

OUTPUT RESULTS:
- CSV file: 'Fluorescence_Intensity_Results.csv'
  Contains: Fluorescence intensity metrics for each analyzed image
  Columns: MeanIntensity, IntegratedDensity, Area, Genotype, Region, Slice, MIP, Stain, Magnification, Counterstain
  Format: One row per analyzed TIFF image
  Metrics: Mean intensity (average pixel value), Integrated density (sum of intensities), Area (pixel count)

Required Libraries and Versions:
- tifffile>=2020.0.0: High-performance TIFF image reading library for loading 16-bit microscopy images (tifffile.imread)
- numpy>=1.20.0: Fundamental package for scientific computing, providing array operations (np.mean, np.sum), mathematical functions, and image data manipulation throughout the analysis pipeline
- pandas>=1.3.0: Data manipulation and analysis library for organizing extracted metrics into structured DataFrames (pd.DataFrame) and exporting results to CSV format (df.to_csv) for statistical analysis
- scipy>=1.7.0: Scientific computing library providing image processing functions including Gaussian filtering (gaussian_filter) for noise reduction and smoothing operations
- scikit-image>=0.18.0: Image processing library implementing thresholding algorithms (threshold_otsu) for automatic image segmentation and restoration functions (rolling_ball) for background subtraction
- os: Standard library for file path operations (os.path.join, os.path.basename), directory creation (os.makedirs), and recursive file searching (os.walk) during batch processing


"""

import os                             # Used for file path operations (os.path.join, os.path.basename), directory creation (os.makedirs), and recursive file searching (os.walk) during batch processing
import tifffile                       # Used for high-performance loading of 16-bit TIFF microscopy images (tifffile.imread) with optimal memory management
import numpy as np                    # Used for array operations (np.mean, np.sum), mathematical functions, image data manipulation, and statistical computations throughout the analysis pipeline
import pandas as pd                   # Used for data manipulation (pd.DataFrame), organizing extracted metrics, and exporting results to CSV format (df.to_csv) for statistical analysis
from scipy.ndimage import gaussian_filter  # Used for Gaussian smoothing (gaussian_filter) to reduce noise and improve image quality for thresholding
from skimage.filters import threshold_otsu  # Used for automatic thresholding (threshold_otsu) to create binary masks for intensity measurements
from skimage import restoration       # Used for background subtraction using rolling ball algorithm (restoration.rolling_ball) to remove uneven illumination

# ---- Parameters ----
ROLLING_BALL_RADIUS = 50      # Radius for rolling ball background subtraction (pixels)
GAUSSIAN_SIGMA = 1.5         # Standard deviation for Gaussian smoothing filter

# ---- Core Functions ----

def calculate_intensity_metrics(image_path, rolling_ball_radius=ROLLING_BALL_RADIUS, gaussian_sigma=GAUSSIAN_SIGMA):
    """
    Calculate intensity metrics for a fluorescence image including background subtraction, smoothing, and thresholding.
    
    Args:
        image_path (str): Path to the input TIFF image file
        rolling_ball_radius (int): Radius for rolling ball background subtraction
        gaussian_sigma (float): Standard deviation for Gaussian smoothing
        
    Returns:
        tuple: (mean_intensity, integrated_density, area) where:
            - mean_intensity: Average pixel intensity above threshold
            - integrated_density: Sum of all pixel intensities above threshold
            - area: Number of pixels above threshold (area in pixels)
    """
    # tifffile: Load 16-bit TIFF image as numpy array for high-performance image processing
    img = tifffile.imread(image_path)

    # scikit-image (restoration): Perform rolling ball background subtraction to remove uneven illumination
    background = restoration.rolling_ball(img, radius=rolling_ball_radius)
    # NumPy: Subtract background and clip negative values to ensure non-negative intensities
    img_bg_subtracted = img - background
    img_bg_subtracted[img_bg_subtracted < 0] = 0

    # scipy (gaussian_filter): Apply Gaussian smoothing to reduce noise and improve thresholding
    img_smoothed = gaussian_filter(img_bg_subtracted, sigma=gaussian_sigma)

    # scikit-image (threshold_otsu): Perform automatic thresholding to create binary mask
    thresh = threshold_otsu(img_smoothed)
    mask = img_smoothed > thresh

    # NumPy: Calculate intensity metrics from thresholded image
    if np.any(mask):
        # NumPy: Calculate mean intensity of pixels above threshold
        mean_intensity = img_smoothed[mask].mean()
        # NumPy: Calculate integrated density (sum of all intensities above threshold)
        integrated_density = img_smoothed[mask].sum()
        # NumPy: Calculate area as number of pixels above threshold
        area = np.count_nonzero(mask)
    else:
        mean_intensity = 0
        integrated_density = 0
        area = 0

    return mean_intensity, integrated_density, area

def parse_filename(filename):
    """
    Parse metadata from fluorescence image filename based on expected naming convention.
    
    Expected format: "Genotype Region Slice MIP Stain Magnification Counterstain_ORG.tif"
    Example: "WT PRS 3 MIP N2A 40X AG_N2A_ORG.tif"
    
    Args:
        filename (str): Input filename to parse
        
    Returns:
        tuple: (genotype, region, slice_number, mip, stain, magnification, counterstain)
    """
    # Python string methods: Extract part before first underscore for metadata parsing
    name_part = filename.split('_')[0]
    # Python string methods: Split by spaces to extract individual metadata components
    tokens = name_part.split()
    
    # Extract metadata with safety checks for array bounds
    genotype = tokens[0] if len(tokens) > 0 else None
    region = tokens[1] if len(tokens) > 1 else None
    slice_number = tokens[2] if len(tokens) > 2 else None
    mip = tokens[3] if len(tokens) > 3 else None
    stain = tokens[4] if len(tokens) > 4 else None
    magnification = tokens[5] if len(tokens) > 5 else None
    counterstain = tokens[6] if len(tokens) > 6 else None
    
    return genotype, region, slice_number, mip, stain, magnification, counterstain

def batch_process(folder_path, output_csv):
    """
    Process multiple fluorescence TIFF images in batch mode, analyzing each image and compiling results.
    
    Args:
        folder_path (str): Directory containing TIFF images to process
        output_csv (str): Path where the results CSV file will be saved
        
    Returns:
        pandas.DataFrame: DataFrame containing metrics for all processed images
    """
    results = []
    
    # os.walk: Recursively find all TIFF files in the folder and subfolders
    tiff_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Filter for organization files, excluding cleaned versions
            if (file.endswith('_ORG.tif') and 
                '_cleaned.tif' not in file):
                tiff_files.append(os.path.join(root, file))
    
    print(f"Found {len(tiff_files)} TIFF files to process...")
    
    for filepath in tiff_files:
        # os.path.basename: Extract filename from full path for processing
        filename = os.path.basename(filepath)
        
        # Determine subunit from filename pattern
        if '_ORG.tif' in filename:
            subunit = 'ORG'  # Generic organization marker
        else:
            continue  # Skip if not organization file
        
        try:
            # Calculate intensity metrics for current image
            mean_intensity, integrated_density, area = calculate_intensity_metrics(filepath)
            # Parse metadata from filename
            genotype, region, slice_number, mip, stain, magnification, counterstain = parse_filename(filename)
            
            # Compile results for current image
            results.append({
                'Filename': filename,
                'Filepath': filepath,
                'Genotype': genotype,
                'Region': region,
                'Slice_Number': slice_number,
                'MIP': mip,
                'Stain': stain,
                'Magnification': magnification,
                'Counterstain': counterstain,
                'Subunit': subunit,
                'Mean_Intensity': mean_intensity,
                'Integrated_Density': integrated_density,
                'Area_pixels': area
            })
            
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    if results:
        # Pandas: Create structured DataFrame from analysis results for easy manipulation
        df = pd.DataFrame(results)
        # Pandas: Export results to CSV file for further analysis
        df.to_csv(output_csv, index=False)
        print(f"\nProcessing complete! {len(results)} files processed successfully.")
        print(f"Results saved to {output_csv}")
        
        # Display comprehensive summary
        print(f"\nSummary:")
        print(f"- Total TIFF files found: {len(tiff_files)}")
        print(f"- Successfully processed: {len(results)}")
        
        # Pandas: Group data by genotype and region for summary statistics
        if 'Genotype' in df.columns and 'Region' in df.columns:
            print(f"\nData breakdown:")
            print(df.groupby(['Genotype', 'Region']).size())
        
        return df
    else:
        print("No files were successfully processed.")
        return None

def main():
    """
    Main function to run the fluorescence intensity analysis.
    Modify the folder path and output file as needed for your setup.
    """
    # Configuration - modify these paths to match your setup
    image_folder = "path/to/your/images"  # Replace with actual path to TIFF files
    output_file = "fluorescence_intensity_results.csv"  # Output CSV filename
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' does not exist.")
        print("Please modify the 'image_folder' variable in the main() function.")
        return
    
    # Process images and get results
    results_df = batch_process(image_folder, output_file)
    
    if results_df is not None:
        print(f"\nAnalysis complete! Processed {len(results_df)} images.")
        print(f"Results saved to '{output_file}'")
        
        # Display additional summary statistics
        if 'Mean_Intensity' in results_df.columns:
            print(f"\nIntensity Statistics:")
            print(f"- Mean intensity range: {results_df['Mean_Intensity'].min():.2f} - {results_df['Mean_Intensity'].max():.2f}")
            print(f"- Integrated density range: {results_df['Integrated_Density'].min():.2f} - {results_df['Integrated_Density'].max():.2f}")
            print(f"- Area range: {results_df['Area_pixels'].min()} - {results_df['Area_pixels'].max()} pixels")

if __name__ == "__main__":
    main()
