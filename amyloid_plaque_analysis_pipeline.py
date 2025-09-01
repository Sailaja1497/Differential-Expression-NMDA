"""
===============================================================================
AMYLOID PLAQUE ANALYSIS PIPELINE
===============================================================================

Author: Sailaja Kuruvada
Date: 2025
Purpose: Comprehensive amyloid plaque analysis and visualization pipeline

This script provides a complete analysis pipeline for amyloid plaque quantification
in brain tissue fluorescence microscopy images. The pipeline includes:
1. Automated plaque detection and segmentation using adaptive thresholding
2. Comprehensive morphological analysis (area, perimeter, circularity, aspect ratio)
3. Plaque load and density calculations
4. Statistical visualization with publication-ready plots
5. Quality control image generation
6. Flexible analysis modes (full processing, plots-only, plaque load focus)

The analysis supports multiple brain regions (PRS, CA1, LEC) with region-specific
optimization parameters and provides both batch processing and targeted analysis modes.

OUTPUT RESULTS:
- CSV file: 'Plaque_Metrics_Summary_new.csv'
  Contains: Comprehensive plaque quantification metrics for each image
  Columns: PlaqueCount, PlaqueLoadPercent, PlaqueDensity, MeanArea, MeanPerimeter, MeanCircularity, MeanAspectRatio
  Metadata: Image names, Genotype, Region, SliceNumber, MIP, Stain, Magnification, Counterstain
  Format: One row per analyzed image
- Quality Control Images: Background-subtracted, binary masks, and overlay images (if enabled)
- Statistical Plots: Publication-ready bar graphs for plaque metrics by region and genotype
- Magnification-specific CSV files: Separate files for 40X and 60X data (plaque load focus mode)

===============================================================================
REQUIRED LIBRARIES AND DEPENDENCIES
===============================================================================

Core Python Libraries:
- os: Operating system interface for file and directory operations
- numpy (np): Numerical computing library for array operations and mathematical functions
- pandas (pd): Data manipulation and analysis library for CSV output and statistical analysis

Image Processing Libraries:
- cv2 (OpenCV): Computer vision library for image reading, preprocessing, and visualization
- skimage: Scikit-image library for advanced image processing algorithms
  - filters: Image filtering and thresholding (Otsu, Li, local adaptive methods)
  - measure: Image measurement and region properties analysis
  - morphology: Morphological operations for noise removal and cleaning
  - exposure: Image intensity rescaling and enhancement
  - filters.threshold_local: Local adaptive thresholding for variable contrast

Data Visualization Libraries:
- matplotlib.pyplot (plt): Primary plotting library for creating publication-quality figures
- seaborn (sns): Statistical data visualization library for enhanced plotting aesthetics

===============================================================================
CONSTANTS AND PARAMETERS
===============================================================================

PIXEL_TO_MICRON: 0.5 - Conversion factor from pixels to micrometers

Background Subtraction:
- BG_SIGMA_DEFAULT: 40 - Gaussian blur sigma for background estimation

Thresholding Parameters:
- THRESHOLD_METHOD_DEFAULT: 'otsu' - Default thresholding method
- THRESHOLD_SCALE_DEFAULT: 1.0 - Multiplicative factor for threshold adjustment

Plaque Filtering:
- PLAQUE_SIZE_MIN_DEFAULT: 5 - Minimum plaque size in pixels
- SAVE_QC_IMAGES: False - Whether to save quality control images

Region-Specific Parameters:
- PRS: Custom parameters for perirhinal cortex
- CA1: Custom parameters for hippocampal CA1 region
- LEC: Custom parameters for lateral entorhinal cortex
  Each region has optimized bg_sigma, threshold_method, threshold_scale, and min_size

===============================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, exposure
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, binary_opening, disk

# ===============================================================================
# CONSTANTS AND CONFIGURATION
# ===============================================================================

# --- User-editable global defaults ---
PIXEL_TO_MICRON = 0.5
BG_SIGMA_DEFAULT = 40
THRESHOLD_METHOD_DEFAULT = 'otsu'
THRESHOLD_SCALE_DEFAULT = 1.0
PLAQUE_SIZE_MIN_DEFAULT = 5
SAVE_QC_IMAGES = False

# --- Region-specific parameters (tweak as needed!) ---
REGION_PARAMS = {
    'PRS':   {'bg_sigma': 40, 'threshold_method': THRESHOLD_METHOD_DEFAULT, 'threshold_scale': THRESHOLD_SCALE_DEFAULT , 'min_size': 5},
    'CA1':   {'bg_sigma': BG_SIGMA_DEFAULT, 'threshold_method': THRESHOLD_METHOD_DEFAULT, 'threshold_scale': THRESHOLD_SCALE_DEFAULT, 'min_size': PLAQUE_SIZE_MIN_DEFAULT},
    'LEC':   {'bg_sigma': BG_SIGMA_DEFAULT, 'threshold_method': THRESHOLD_METHOD_DEFAULT, 'threshold_scale': THRESHOLD_SCALE_DEFAULT, 'min_size': PLAQUE_SIZE_MIN_DEFAULT}
}

# ===============================================================================
# IMAGE PROCESSING FUNCTIONS
# ===============================================================================

def load_image(image_path):
    """
    Load fluorescence microscopy image as grayscale.
    
    Parameters:
    -----------
    image_path : str
        File path to the image file
        
    Returns:
    --------
    numpy.ndarray
        Grayscale image array (uint8 or uint16 depending on source)
        
    Notes:
    ------
    - Uses OpenCV's IMREAD_GRAYSCALE for consistent grayscale loading
    - Maintains original bit depth from source file
    """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def preprocess_image(img, bg_sigma):
    """
    Preprocess fluorescence image for plaque segmentation.
    
    This function applies a series of preprocessing steps to enhance
    plaque visibility and reduce noise:
    1. Gaussian blur for noise reduction
    2. Background estimation using large Gaussian kernel
    3. Background subtraction to enhance contrast
    4. Intensity rescaling to full dynamic range
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input fluorescence image
    bg_sigma : float
        Sigma parameter for background estimation Gaussian blur
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed image ready for segmentation
        
    Notes:
    ------
    - Uses 5x5 Gaussian blur for initial denoising
    - Background estimation uses large kernel (bg_sigma)
    - Final output is rescaled to 0-255 range
    """
    # Apply Gaussian blur for noise reduction
    denoised = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Estimate background using large Gaussian kernel
    background = cv2.GaussianBlur(denoised, (0, 0), bg_sigma)
    
    # Subtract background to enhance contrast
    subtracted = cv2.subtract(denoised, background)
    
    # Rescale intensity to full dynamic range
    prepped = exposure.rescale_intensity(subtracted, out_range=(0, 255)).astype(np.uint8)
    
    return prepped


def segment_plaques(img, threshold_method, threshold_scale, min_size):
    """
    Segment amyloid plaques from preprocessed fluorescence images.
    
    This function implements a multi-step segmentation pipeline optimized
    for amyloid plaque detection:
    1. Adaptive thresholding using specified method
    2. Small object removal for noise reduction
    3. Hole filling for complete plaque regions
    4. Boundary smoothing for refined segmentation
    
    Parameters:
    -----------
    img : numpy.ndarray
        Preprocessed fluorescence image
    threshold_method : str
        Thresholding method ('otsu', 'li', or 'local')
    threshold_scale : float
        Multiplicative factor to adjust threshold sensitivity
    min_size : int
        Minimum size in pixels for valid plaque objects
        
    Returns:
    --------
    tuple
        (binary_mask, labeled_image) where:
        - binary_mask: Boolean array of segmented plaques
        - labeled_image: Integer array with unique labels for each plaque
        
    Notes:
    ------
    - Otsu thresholding: Automatic threshold selection for bimodal distributions
    - Li thresholding: Iterative threshold selection method
    - Local thresholding: Adaptive thresholding using local image statistics
    - Uses morphological operations for post-processing refinement
    """
    # Step 1: Apply thresholding based on specified method
    if threshold_method == 'otsu':
        threshold = filters.threshold_otsu(img) * threshold_scale
        binary = img > threshold
    elif threshold_method == 'li':
        threshold = filters.threshold_li(img) * threshold_scale
        binary = img > threshold
    elif threshold_method == 'local':
        local_thresh = threshold_local(img, block_size=151, method='gaussian')
        binary = img > local_thresh
    else:
        raise ValueError('Unknown threshold_method')
    
    # Step 2: Remove small speckles
    cleaned = remove_small_objects(binary, min_size)
    
    # Step 3: Fill holes (area up to 30 px)
    cleaned = remove_small_holes(cleaned, area_threshold=30)
    
    # Step 4: Smooth boundaries with closing
    cleaned = binary_closing(cleaned, disk(2))
    
    return cleaned, measure.label(cleaned)


# ===============================================================================
# METADATA AND FILE PROCESSING FUNCTIONS
# ===============================================================================

def parse_metadata(filename):
    """
    Parse experimental metadata from fluorescence microscopy filenames.
    
    Extracts experimental parameters from standardized filename format:
    'Genotype Region SliceNumber MIP Stain Magnification Counterstain.TIF'
    
    Example: 'WT PRS 3 MIP N2A 40X AG_AG_ORG.TIF'
    
    Parameters:
    -----------
    filename : str
        Input filename to parse
        
    Returns:
    --------
    dict
        Dictionary containing parsed metadata fields:
        - Genotype: Mouse genotype (e.g., 'WT', 'APP')
        - Region: Brain region (e.g., 'PRS', 'CA1', 'LEC')
        - SliceNumber: Tissue slice identifier
        - MIP: Maximum intensity projection indicator
        - Stain: Fluorescent marker channel
        - Magnification: Microscope magnification
        - Counterstain: Secondary staining identifier
    """
    # Remove file extension and split by spaces
    tokens = filename.replace('.TIF', '').split(' ')
    
    # Extract metadata with safe indexing
    return {
        'Genotype': tokens[0] if len(tokens) > 0 else 'Unknown',
        'Region': tokens[1] if len(tokens) > 1 else 'Unknown',
        'SliceNumber': tokens[2] if len(tokens) > 2 else 'Unknown',
        'MIP': tokens[3] if len(tokens) > 3 else 'Unknown',
        'Stain': tokens[4] if len(tokens) > 4 else 'Unknown',
        'Magnification': tokens[5] if len(tokens) > 5 else 'Unknown',
        'Counterstain': tokens[6] if len(tokens) > 6 else 'Unknown'
    }


# ===============================================================================
# QUANTIFICATION AND ANALYSIS FUNCTIONS
# ===============================================================================

def quantify_plaques_summary(labeled, image_shape):
    """
    Calculate comprehensive plaque morphology and distribution metrics.
    
    This function computes a complete set of quantitative measures for
    amyloid plaque analysis including:
    1. Basic counts and areas
    2. Morphological properties (circularity, aspect ratio)
    3. Distribution metrics (load percentage, density)
    4. Statistical summaries of plaque populations
    
    Parameters:
    -----------
    labeled : numpy.ndarray
        Labeled image where each plaque has a unique integer label
    image_shape : tuple
        Shape of original image (height, width)
        
    Returns:
    --------
    dict
        Dictionary containing all plaque metrics:
        - PlaqueCount: Total number of plaques detected
        - PlaqueLoadPercent: Percentage of image area occupied by plaques
        - PlaqueDensity: Number of plaques per square millimeter
        - MeanArea: Average plaque area in square micrometers
        - MeanPerimeter: Average plaque perimeter in micrometers
        - MeanCircularity: Average circularity (0-1, 1 = perfect circle)
        - MeanAspectRatio: Average aspect ratio (major/minor axis)
        
    Notes:
    ------
    - All area measurements converted to square micrometers
    - Circularity calculated as (4π × area) / (perimeter²)
    - Aspect ratio calculated as major_axis_length / minor_axis_length
    - Handles edge cases (zero perimeter, zero minor axis) gracefully
    """
    # Extract region properties for all plaques
    props = measure.regionprops(labeled)
    
    # Calculate basic measurements
    areas = [p.area for p in props]
    perimeters = [p.perimeter for p in props]
    
    # Calculate derived morphological properties
    circularities = [(4 * np.pi * p.area) / (p.perimeter ** 2) if p.perimeter != 0 else 0 for p in props]
    aspect_ratios = [p.major_axis_length / p.minor_axis_length if p.minor_axis_length != 0 else 0 for p in props]

    # Calculate area-based metrics
    plaque_area = np.sum(areas) * (PIXEL_TO_MICRON ** 2)
    image_area = image_shape[0] * image_shape[1] * (PIXEL_TO_MICRON ** 2)
    plaque_load_percent = (plaque_area / image_area) * 100
    plaque_density = len(props) / image_area

    # Compile results
    return {
        'PlaqueCount': len(props),
        'PlaqueLoadPercent': plaque_load_percent,
        'PlaqueDensity': plaque_density,
        'MeanArea': np.mean(areas) if areas else np.nan,
        'MeanPerimeter': np.mean(perimeters) if perimeters else np.nan,
        'MeanCircularity': np.mean(circularities) if circularities else np.nan,
        'MeanAspectRatio': np.mean(aspect_ratios) if aspect_ratios else np.nan
    }


def process_amyloid_image(image_path, output_folder=None):
    """
    Process a single amyloid image through the complete analysis pipeline.
    
    This function orchestrates the entire analysis workflow for one image:
    1. Loads and preprocesses the image
    2. Applies region-specific segmentation parameters
    3. Quantifies plaque metrics
    4. Saves quality control images (optional)
    5. Returns comprehensive results with metadata
    
    Parameters:
    -----------
    image_path : str
        File path to the amyloid image file
    output_folder : str, optional
        Directory to save quality control images
        
    Returns:
    --------
    dict
        Dictionary containing all analysis results and metadata
        
    Notes:
    ------
    - Uses region-specific parameters from REGION_PARAMS
    - Falls back to default parameters if region not found
    - Quality control images include background-subtracted, mask, and overlay
    - Overlay shows original image with plaque outlines in red
    """
    # Extract metadata from filename
    filename = os.path.basename(image_path)
    metadata = parse_metadata(filename)
    region = metadata['Region']
    
    # Pick parameters for this region or fallback to global defaults
    params = REGION_PARAMS.get(region, {
        'bg_sigma': BG_SIGMA_DEFAULT,
        'threshold_method': THRESHOLD_METHOD_DEFAULT,
        'threshold_scale': THRESHOLD_SCALE_DEFAULT,
        'min_size': PLAQUE_SIZE_MIN_DEFAULT
    })
    
    # Load and process image
    raw_img = load_image(image_path)
    prepped = preprocess_image(raw_img, bg_sigma=params['bg_sigma'])
    
    # Segment plaques
    mask, plaques = segment_plaques(
        prepped, 
        threshold_method=params['threshold_method'],
        threshold_scale=params['threshold_scale'],
        min_size=params['min_size']
    )
    
    # Calculate metrics
    summary_metrics = quantify_plaques_summary(plaques, raw_img.shape)
    summary_metrics.update(metadata)
    summary_metrics['Image'] = filename

    # Save quality control images if requested
    if output_folder and SAVE_QC_IMAGES:
        # Save background-subtracted image
        cv2.imwrite(os.path.join(output_folder, filename.replace('.TIF', '_BGsub.tif')), prepped)
        
        # Save binary mask
        mask_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, filename.replace('.TIF', '_mask.tif')), mask_vis)
        
        # Save overlay image (original with plaque outlines)
        overlay = cv2.cvtColor(prepped, cv2.COLOR_GRAY2BGR)
        overlay[mask] = [0, 0, 255]  # Red outline for plaques
        cv2.imwrite(os.path.join(output_folder, filename.replace('.TIF', '_overlay.tif')), overlay)
    
    return summary_metrics


# ===============================================================================
# VISUALIZATION FUNCTIONS
# ===============================================================================

def generate_bargraphs(df, output_folder):
    """
    Generate publication-quality bar graphs for statistical analysis.
    
    This function creates comprehensive visualizations comparing plaque
    metrics across different experimental conditions:
    1. Plaque count by region and genotype
    2. Plaque load percentage by region and genotype
    3. Plaque density by region and genotype
    4. Separate plots for different magnifications
    5. Combined plots for overall analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing all plaque metrics and metadata
    output_folder : str
        Directory to save generated plots
        
    Notes:
    ------
    - Uses seaborn for enhanced plotting aesthetics
    - Creates separate plots for 40X and 60X magnifications
    - Generates combined plots for overall analysis
    - Uses consistent color scheme (light blue for WT, dark blue for APP)
    - Includes error bars (standard error) for statistical significance
    - Saves high-resolution (300 DPI) publication-ready figures
    """
    import seaborn as sns
    sns.set(style="whitegrid")

    def plot_metric(metric, ylabel, title, fname, data_subset):
        """Helper function to create individual metric plots."""
        plt.figure(figsize=(10, 7))
        
        # Create bar plot only (no individual data points)
        sns.barplot(data=data_subset, x='Region', y=metric, hue='Genotype', 
                   palette={'WT': '#ADD8E6', 'APP': '#00008B'}, 
                   errorbar='se', capsize=0.1, alpha=0.8)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel("Region", fontsize=12)
        plt.legend(title="Genotype", title_fontsize=12, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, fname), dpi=300, bbox_inches='tight')
        plt.close()

    # Separate plots for 40X and 60X
    for magnification in ['40X', '60X']:
        # Filter data for current magnification
        mag_data = df[df['Magnification'] == magnification]
        
        if len(mag_data) > 0:
            print(f"Creating plots for {magnification} magnification ({len(mag_data)} samples)")
            
            # Plaque count plot
            plot_metric('PlaqueCount', 'Average Plaque Count', 
                       f'Plaque Count by Region and Genotype ({magnification})', 
                       f'PlaqueCount_by_Region_Genotype_{magnification}.png', mag_data)
            
            # Plaque load plot
            plot_metric('PlaqueLoadPercent', 'Plaque Load (%)', 
                       f'Plaque Load by Region and Genotype ({magnification})', 
                       f'PlaqueLoad_by_Region_Genotype_{magnification}.png', mag_data)
            
            # Plaque density plot
            plot_metric('PlaqueDensity', 'Plaque Density (plaques/mm²)', 
                       f'Plaque Density by Region and Genotype ({magnification})', 
                       f'PlaqueDensity_by_Region_Genotype_{magnification}.png', mag_data)
        else:
            print(f"No data found for {magnification} magnification")
    
    # Also create combined plots (all magnifications)
    print(f"Creating combined plots (all magnifications, {len(df)} samples)")
    
    plot_metric('PlaqueCount', 'Average Plaque Count', 
               'Plaque Count by Region and Genotype (All Magnifications)', 
               'PlaqueCount_by_Region_Genotype_All.png', df)
    
    plot_metric('PlaqueLoadPercent', 'Plaque Load (%)', 
               'Plaque Load by Region and Genotype (All Magnifications)', 
               'PlaqueLoad_by_Region_Genotype_All.png', df)
    
    plot_metric('PlaqueDensity', 'Plaque Density (plaques/mm²)', 
               'Plaque Density by Region and Genotype (All Magnifications)', 
               'PlaqueDensity_by_Region_Genotype_All.png', df)


# ===============================================================================
# BATCH PROCESSING FUNCTIONS
# ===============================================================================

def batch_process_amyloid_images(image_folder, output_folder):
    """
    Execute complete batch analysis of all amyloid images in directory.
    
    This function processes all amyloid images in the specified directory
    and its subdirectories:
    1. Discovers all AG_AG_ORG.TIF files recursively
    2. Processes each image through the complete analysis pipeline
    3. Compiles results into comprehensive CSV report
    4. Generates statistical visualizations
    5. Saves quality control images for validation
    
    Parameters:
    -----------
    image_folder : str
        Root directory containing fluorescence microscopy images
    output_folder : str
        Directory to save analysis results and visualizations
        
    Notes:
    ------
    - Recursively searches all subdirectories for amyloid images
    - Only processes files ending with 'AG_AG_ORG.TIF'
    - Creates output directory if it doesn't exist
    - Handles errors gracefully and continues processing
    - Generates empty CSV with headers if no files processed
    - Creates visualizations only if sufficient data available
    """
    all_summaries = []
    os.makedirs(output_folder, exist_ok=True)
    processed_files = 0
    
    # Walk through directory tree to find all amyloid images
    for root, dirs, files in os.walk(image_folder):
        for fname in files:
            if fname.upper().endswith('AG_AG_ORG.TIF'):
                img_path = os.path.join(root, fname)
                print(f"Processing: {img_path}")
                
                try:
                    # Process image and collect results
                    summary = process_amyloid_image(img_path, output_folder=output_folder)
                    all_summaries.append(summary)
                    processed_files += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    print(f"Total files processed: {processed_files}")

    # Create DataFrame and save results
    if not all_summaries:
        print("No files were processed. Creating empty CSV with headers.")
        df = pd.DataFrame(columns=[
            'Image', 'Genotype', 'Region', 'SliceNumber', 'MIP', 'Stain', 
            'Magnification', 'Counterstain', 'PlaqueCount', 'PlaqueLoadPercent', 
            'PlaqueDensity', 'MeanArea', 'MeanPerimeter', 'MeanCircularity', 'MeanAspectRatio'
        ])
    else:
        df = pd.DataFrame(all_summaries)
    
    # Save results to CSV
    output_path = os.path.join(output_folder, 'Plaque_Metrics_Summary_new.csv')
    df.to_csv(output_path, index=False)
    print(f'📊 Plaque metrics saved to: {output_path}')
    
    # Generate visualizations if sufficient data available
    if len(df) > 0 and 'Genotype' in df.columns and 'Region' in df.columns:
        generate_bargraphs(df, output_folder)
        print(f'📈 Bar graphs saved in: {output_folder}')
    else:
        print("No data to plot or missing required columns.")


# ===============================================================================
# SPECIALIZED ANALYSIS FUNCTIONS
# ===============================================================================

def create_plots_from_existing_csv(csv_path, output_folder):
    """
    Create plots from existing CSV file without processing images.
    
    This function allows for quick visualization of previously processed data
    without re-running the entire image analysis pipeline. Useful for:
    - Recreating plots with different parameters
    - Generating additional visualizations
    - Quick data exploration and validation
    
    Parameters:
    -----------
    csv_path : str
        Path to existing CSV file containing plaque metrics
    output_folder : str
        Directory to save generated plots
        
    Notes:
    ------
    - Reads existing CSV file and validates data structure
    - Generates all standard visualizations (count, load, density)
    - Provides detailed feedback on data loading and processing
    - Handles missing files gracefully with informative error messages
    """
    print(f"=== Creating Plots from Existing CSV ===")
    print(f"Reading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Available magnifications: {df['Magnification'].unique()}")
    
    # Generate plots
    generate_bargraphs(df, output_folder)
    print(f"✅ Plots saved to: {output_folder}")


def plot_plaque_load_only(csv_path, output_folder):
    """
    Create plots for plaque load only from existing CSV file.
    
    This specialized function focuses exclusively on plaque load analysis,
    providing detailed visualizations and data exports for this key metric.
    Useful for focused analysis and publication preparation.
    
    Parameters:
    -----------
    csv_path : str
        Path to existing CSV file containing plaque metrics
    output_folder : str
        Directory to save generated plots and CSV files
        
    Notes:
    ------
    - Creates plaque load plots for each magnification separately
    - Generates combined plaque load plot for all magnifications
    - Exports separate CSV files for each magnification
    - Uses consistent color scheme and formatting
    - Provides comprehensive feedback on processing steps
    """
    import seaborn as sns
    
    print(f"=== Creating Plaque Load Plots Only ===")
    print(f"Reading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Available magnifications: {df['Magnification'].unique()}")
    
    # Set up plotting style
    sns.set(style="whitegrid")
    
    def plot_plaque_load(data_subset, magnification_label, filename):
        """Helper function to create plaque load plots."""
        plt.figure(figsize=(10, 7))
        
        # Create bar plot only (no individual data points)
        sns.barplot(data=data_subset, x='Region', y='PlaqueLoadPercent', hue='Genotype', 
                   palette={'WT': '#ADD8E6', 'APP': '#00008B'}, 
                   errorbar='se', capsize=0.1, alpha=0.8)
        
        plt.title(f'Plaque Load by Region and Genotype ({magnification_label})', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Plaque Load (%)', fontsize=12)
        plt.xlabel("Region", fontsize=12)
        plt.legend(title="Genotype", title_fontsize=12, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Separate plots for each magnification
    for magnification in df['Magnification'].unique():
        # Filter data for current magnification
        mag_data = df[df['Magnification'] == magnification]
        
        if len(mag_data) > 0:
            print(f"\nCreating plaque load plot for {magnification} magnification ({len(mag_data)} samples)")
            
            plot_plaque_load(mag_data, magnification, f'PlaqueLoad_by_Region_Genotype_{magnification}.png')
        else:
            print(f"No data found for {magnification} magnification")
    
    # Also create combined plot (all magnifications)
    print(f"\nCreating combined plaque load plot (all magnifications, {len(df)} samples)")
    plot_plaque_load(df, "All Magnifications", 'PlaqueLoad_by_Region_Genotype_All.png')
    
    # Create separate CSV files for each magnification
    print(f"\n=== Creating Separate CSV Files ===")
    for magnification in df['Magnification'].unique():
        mag_data = df[df['Magnification'] == magnification]
        if len(mag_data) > 0:
            csv_filename = f'Plaque_Metrics_Summary_{magnification}.csv'
            csv_path = os.path.join(output_folder, csv_filename)
            mag_data.to_csv(csv_path, index=False)
            print(f"Saved {magnification} data: {csv_filename} ({len(mag_data)} samples)")
    
    print(f"\n✅ All plaque load plots and CSV files saved to: {output_folder}")


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

if __name__ == "__main__":
    """
    Main execution block for amyloid plaque analysis pipeline.
    
    Sets up the analysis pipeline and processes all amyloid images in the
    specified directory. This script is designed for batch processing of
    large datasets containing multiple experimental conditions and brain regions.
    
    Usage:
    ------
    python amyloid_plaque_analysis_pipeline.py
    
    Output:
    -------
    - CSV file containing comprehensive plaque metrics
    - Quality control images for validation (if enabled)
    - Statistical visualizations
    
    Alternative Modes:
    ------------------
    Uncomment the lines below to run specialized analysis modes:
    - create_plots_from_existing_csv(): Generate plots from existing data
    - plot_plaque_load_only(): Focus on plaque load analysis only
    """
    print("=" * 80)
    print("AMYLOID PLAQUE ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Define input and output directories
    image_folder = "IMAGE_ANALYSIS_FOLDER"  # Replace with your image folder path
    output_folder = "OUTPUT_RESULTS_FOLDER"  # Replace with your output folder path
    
    print(f"📁 Input directory: {image_folder}")
    print(f"📂 Output directory: {output_folder}")
    print("🚀 Starting batch analysis...")
    print("=" * 80)
    
    # Execute batch analysis
    batch_process_amyloid_images(image_folder, output_folder)
    
    print("=" * 80)
    print("🎉 Amyloid plaque analysis pipeline completed successfully!")
    print("=" * 80)

# ===============================================================================
# ALTERNATIVE EXECUTION MODES
# ===============================================================================

# Uncomment the line below to run plots-only mode
# create_plots_from_existing_csv("path/to/your/Plaque_Metrics_Summary.csv", output_folder)

# Uncomment the line below to run plaque load plots only
# plot_plaque_load_only("path/to/your/Plaque_Metrics_Summary_40X.csv", output_folder)
