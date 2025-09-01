"""

Author: Sailaja Kuruvada
Date: 2025

Puncta Clustering Analysis Tool

This script analyzes fluorescent microscopy images to detect and cluster puncta (small fluorescent spots).
It performs blob detection using Laplacian of Gaussian (LoG) and clusters the detected puncta using DBSCAN.
The tool extracts various metrics including puncta counts, cluster statistics, and spatial distribution data.

Key Features:
- Automatic puncta detection using blob detection algorithms
- Spatial clustering of puncta using DBSCAN
- Comprehensive metrics extraction for analysis
- Batch processing of multiple images
- Visualization of clustering results
- Metadata parsing from filenames

OUTPUT RESULTS:
- CSV file: 'Puncta_Clustering_Results.csv'
  Contains: Puncta clustering analysis results for each image
  Columns: PunctaCount, ClusterCount, MeanClusterSize, MaxClusterSize, TotalClusterArea, MeanPunctaArea, Genotype, Region, Slice, MIP, Stain, Magnification
  Format: One row per analyzed image
  Metrics: Puncta counts, cluster statistics, spatial distribution data
- Visualization Images: Cluster analysis plots showing detected puncta and cluster assignments

Required Libraries and Versions:
- opencv-python>=4.5.0: Computer vision library for image processing operations including convex hull calculations (cv2.convexHull) and contour area measurements (cv2.contourArea) to analyze cluster spatial properties
- numpy>=1.20.0: Fundamental package for scientific computing, providing multi-dimensional array objects and mathematical functions for coordinate manipulation (coordinates[:, :2]), statistical calculations (np.mean, np.sum), and mathematical operations throughout the analysis pipeline
- pandas>=1.3.0: Data manipulation and analysis library for organizing extracted metrics into structured DataFrames (pd.DataFrame) and exporting results to CSV format (df.to_csv) for statistical analysis
- matplotlib>=3.3.0: Comprehensive plotting library for creating publication-quality visualizations of detected puncta clusters with color-coded cluster assignments, including scatter plots (plt.scatter), image display (plt.imshow), and saving cluster analysis plots (plt.savefig)
- scikit-image>=0.18.0: Image processing library implementing blob detection algorithms (Laplacian of Gaussian) for automatic puncta identification (blob_log) and image I/O operations (io.imread) for loading microscopy images
- scikit-learn>=1.0.0: Machine learning library providing the DBSCAN clustering algorithm (DBSCAN.fit) for spatial grouping of detected puncta based on proximity metrics to group nearby fluorescent spots into clusters

"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from sklearn.cluster import DBSCAN
from skimage import io
import os

# ---- Parameters ----
BLOB_THRESHOLD = 0.02      # Threshold for blob detection sensitivity
DBSCAN_EPS = 5             # Maximum distance between points to be considered neighbors
DBSCAN_MIN_SAMPLES = 3     # Minimum number of points required to form a cluster

# ---- Core Functions ----

def load_image(image_path):
    """
    Load and preprocess an image for analysis.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        numpy.ndarray: Normalized grayscale image (0-1 range)
    """
    img = io.imread(image_path, as_gray=True)
    img = (img / np.max(img))  # Normalize to 0-1 range
    return img

def detect_puncta(image):
    """
    Detect puncta (fluorescent spots) in the image using Laplacian of Gaussian blob detection.
    
    Args:
        image (numpy.ndarray): Input grayscale image
        
    Returns:
        tuple: (coordinates, areas) where coordinates are (y, x) positions and areas are puncta areas
    """
    blobs = blob_log(image, max_sigma=5, num_sigma=10, threshold=BLOB_THRESHOLD)
    coordinates = blobs[:, :2]  # Extract y, x coordinates
    radii = blobs[:, 2] * np.sqrt(2)  # Calculate approximate radius of each blob
    areas = np.pi * (radii ** 2)      # Calculate area of each puncta
    return coordinates, areas

def cluster_puncta(coordinates):
    """
    Cluster detected puncta using DBSCAN algorithm based on spatial proximity.
    
    Args:
        coordinates (numpy.ndarray): Array of (y, x) coordinates for detected puncta
        
    Returns:
        tuple: (labels, clustering_model) where labels are cluster assignments and clustering_model is the fitted DBSCAN model
    """
    if len(coordinates) == 0:
        return np.array([]), None
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(coordinates)
    return clustering.labels_, clustering

def visualize_clusters(image, coordinates, labels, save_path):
    """
    Create and save a visualization of the detected puncta and their clustering results.
    
    Args:
        image (numpy.ndarray): Input image for background
        coordinates (numpy.ndarray): Puncta coordinates
        labels (numpy.ndarray): Cluster labels for each puncta
        save_path (str): Path where the visualization image will be saved
    """
    plt.figure(figsize=(8,8))
    plt.imshow(image, cmap='gray')
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:
            color = 'k'  # Black for noise points
            size = 30
        else:
            color = plt.cm.nipy_spectral(float(label) / len(unique_labels))
            size = 50
        points = coordinates[labels == label]
        plt.scatter(points[:,1], points[:,0], s=size, c=[color], label=f'Cluster {label}')
    
    plt.axis('off')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def parse_filename_metadata(base_name):
    """
    Parse metadata from image filename based on expected naming convention.
    
    Expected format: "Genotype Region Slice Image Stain Magnification Counterstain"
    
    Args:
        base_name (str): Filename without extension
        
    Returns:
        dict: Dictionary containing parsed metadata fields
    """
    parts = base_name.split(' ')
    return {
        'Genotype': parts[0],
        'Region': parts[1],
        'Slice': parts[2],
        'Image': parts[3],
        'Stain': parts[4],
        'Magnification': parts[5],
        'Counterstain': parts[6] if len(parts) > 6 else ''
    }

def analyze_image(image_path, clustering_dir):
    """
    Perform complete analysis of a single image: detect puncta, cluster them, and extract metrics.
    
    Args:
        image_path (str): Path to the input image
        clustering_dir (str): Directory to save clustering visualizations
        
    Returns:
        dict: Dictionary containing all extracted metrics and metadata
    """
    os.makedirs(clustering_dir, exist_ok=True)

    # Load and process image
    img = load_image(image_path)
    coordinates, puncta_areas = detect_puncta(img)
    labels, clustering = cluster_puncta(coordinates)

    # Generate visualization
    base_name = os.path.basename(image_path).replace('.tif', '')
    cluster_image_path = os.path.join(clustering_dir, f'{base_name}_clusters.png')
    visualize_clusters(img, coordinates, labels, cluster_image_path)

    # Parse metadata from filename
    metadata = parse_filename_metadata(base_name)

    # Calculate basic statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    total_puncta = len(coordinates)
    total_puncta_area = np.sum(puncta_areas)

    # Calculate density metrics
    image_area = img.shape[0] * img.shape[1]
    puncta_density = total_puncta / image_area if image_area > 0 else 0
    avg_puncta_per_cluster = (total_puncta - n_noise) / n_clusters if n_clusters > 0 else 0

    # Calculate cluster-specific metrics
    cluster_areas = []
    cluster_densities = []
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    for label in unique_labels:
        points = coordinates[labels == label]
        if len(points) > 2:
            hull = cv2.convexHull(points.astype(np.float32))
            area = cv2.contourArea(hull)
            cluster_areas.append(area)
            density = len(points) / area if area > 0 else 0
            cluster_densities.append(density)

    avg_cluster_area = np.mean(cluster_areas) if cluster_areas else 0
    avg_cluster_density = np.mean(cluster_densities) if cluster_densities else 0
    total_cluster_area = np.sum(cluster_areas) if cluster_areas else 0

    # Compile all metrics
    metrics = {
        'Image_File': base_name,
        'Genotype': metadata['Genotype'],
        'Region': metadata['Region'],
        'Slice': metadata['Slice'],
        'Image': metadata['Image'],
        'Stain': metadata['Stain'],
        'Magnification': metadata['Magnification'],
        'Counterstain': metadata['Counterstain'],
        'Marker': metadata['Stain'],
        'Total_Puncta': total_puncta,
        'Total_Puncta_Area': total_puncta_area,
        'Puncta_Density': puncta_density,
        'N_Clusters': n_clusters,
        'Noise_Points': n_noise,
        'Avg_Puncta_Per_Cluster': avg_puncta_per_cluster,
        'Avg_Cluster_Area': avg_cluster_area,
        'Avg_Cluster_Density': avg_cluster_density,
        'Total_Cluster_Area': total_cluster_area
    }
    return metrics

# ---- Batch Processing Functions ----

def batch_process(image_dir, output_dir=None):
    """
    Process multiple images in batch mode, analyzing each image and compiling results.
    
    Args:
        image_dir (str): Directory containing images to process
        output_dir (str, optional): Directory to save results. If None, creates 'Clustering Analysis' subdirectory
        
    Returns:
        pandas.DataFrame: DataFrame containing metrics for all processed images
    """
    if output_dir is None:
        clustering_dir = os.path.join(image_dir, 'Clustering Analysis')
    else:
        clustering_dir = output_dir
    
    os.makedirs(clustering_dir, exist_ok=True)

    results = []
    
    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(image_dir):
        for fname in files:
            if fname.endswith('.tif') and '_N2B_ORG_cleaned' in fname:
                img_path = os.path.join(root, fname)
                try:
                    metrics = analyze_image(img_path, clustering_dir)
                    results.append(metrics)
                    print(f"Processed: {fname}")
                except Exception as e:
                    print(f"Error processing {fname}: {str(e)}")
    
    if not results:
        print("No matching files found or all files failed to process.")
        return None
    
    df = pd.DataFrame(results)
    
    # Ensure columns are in the expected order
    expected_columns = ['Image_File', 'Genotype', 'Region', 'Slice', 'Image', 'Stain',
                       'Magnification', 'Counterstain', 'Marker',
                       'Total_Puncta', 'Total_Puncta_Area', 'Puncta_Density',
                       'N_Clusters', 'Noise_Points', 'Avg_Puncta_Per_Cluster',
                       'Avg_Cluster_Area', 'Avg_Cluster_Density', 'Total_Cluster_Area']
    
    available_columns = [col for col in expected_columns if col in df.columns]
    df = df[available_columns]
    
    # Save results
    if output_dir is None:
        output_dir = clustering_dir
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Cluster_Metrics_AllMarkers_nba.csv')
    df.to_csv(output_file, index=False)
    
    print(f'Batch processing complete. Processed {len(results)} files.')
    print(f'Outputs saved in {output_dir}')
    
    return df

def main():
    """
    Main function to run the puncta clustering analysis.
    Modify the image directory path as needed for your setup.
    """
    # Example usage - modify this path to match your image directory
    image_directory = "path/to/your/images"  # Replace with actual path
    
    if not os.path.exists(image_directory):
        print(f"Error: Image directory '{image_directory}' does not exist.")
        print("Please modify the 'image_directory' variable in the main() function.")
        return
    
    # Process images and get results
    results_df = batch_process(image_directory)
    
    if results_df is not None:
        print(f"\nAnalysis complete! Processed {len(results_df)} images.")
        print("Results saved to 'Cluster_Metrics_AllMarkers_nba.csv'")

if __name__ == "__main__":
    main()
