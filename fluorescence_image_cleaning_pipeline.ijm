// ===============================================================================
// FLUORESCENCE IMAGE CLEANING PIPELINE
// ===============================================================================
//
// Author: Sailaja Kuruvada
// Date: 2025
// Purpose: Automated background subtraction and cleaning of fluorescence microscopy images
//
// This ImageJ macro performs batch processing of fluorescence microscopy images to:
// 1. Remove uneven background illumination using Gaussian blur subtraction
// 2. Process N2A, N2B, and PSD95 channel images from multiple subfolders
// 3. Convert images to 16-bit for enhanced dynamic range
// 4. Apply background correction using large Gaussian kernel (sigma=80)
// 5. Save cleaned images with standardized naming convention
//
// INPUT REQUIREMENTS:
// - Main folder: "Image Analysis New" containing subfolders with fluorescence images
// - File naming convention: "*_N2A_ORG.tif", "*_N2B_ORG.tif", "*_PSD_ORG.tif"
// - Image format: TIFF files with original fluorescence data
//
// OUTPUT RESULTS:
// - Cleaned images: "*_ORG_cleaned.tif" files saved in same subfolders as originals
// - Background-corrected fluorescence images ready for quantitative analysis
// - 16-bit TIFF format for optimal dynamic range preservation
//
// PROCESSING PARAMETERS:
// - Gaussian blur sigma: 80 pixels (optimized for background estimation)
// - Image bit depth: 16-bit (enhanced dynamic range)
// - Background subtraction: Original - Blurred (removes uneven illumination)
// - File naming: Preserves original "_ORG" suffix, adds "_cleaned" suffix
//
// ===============================================================================
// REQUIRED IMAGEJ FUNCTIONS AND OPERATIONS
// ===============================================================================
//
// Core ImageJ Operations:
// - getDirectory(): User interface for folder selection
// - getFileList(): Directory listing and file discovery
// - File.isDirectory(): Directory validation and recursive processing
// - endsWith(): File type filtering for specific channel markers
//
// Image Processing Operations:
// - open(): Image loading and display
// - run("16-bit"): Bit depth conversion for enhanced dynamic range
// - run("Duplicate"): Image duplication for processing pipeline
// - run("Gaussian Blur"): Background estimation using large kernel
// - imageCalculator(): Background subtraction (Original - Blurred)
//
// File Management Operations:
// - getTitle(): Image window title retrieval
// - selectImage(): Active image window selection
// - rename(): Image window title modification
// - saveAs("Tiff"): High-quality TIFF file export
// - close(): Memory management and cleanup
//
// ===============================================================================

// Main execution block - initiates the batch processing pipeline
mainDir = getDirectory("Choose the main folder (Image Analysis New)");
processFolder(mainDir);

// ===============================================================================
// CORE PROCESSING FUNCTION
// ===============================================================================
//
// Function: processFolder(dir)
// Purpose: Recursively processes all subfolders and applies cleaning to fluorescence images
//
// Parameters:
// - dir: String path to current directory being processed
//
// Processing Logic:
// 1. Directory scanning: Lists all files and subdirectories
// 2. Recursive processing: Calls itself for subdirectories
// 3. File filtering: Identifies N2A, N2B, and PSD95 channel images
// 4. Image cleaning: Applies background subtraction pipeline
// 5. File saving: Exports cleaned images with standardized naming
//
// Channel Detection:
// - N2A: NR2A subunit of NMDA receptors (synaptic marker)
// - N2B: NR2B subunit of NMDA receptors (synaptic marker)  
// - PSD: Postsynaptic density protein 95 (synaptic scaffold marker)
//
// ===============================================================================

function processFolder(dir) {
    // Get list of all files and directories in current folder
    list = getFileList(dir);
    
    // Iterate through each item in the directory
    for (i = 0; i < list.length; i++) {
        
        // ===============================================================================
        // RECURSIVE DIRECTORY PROCESSING
        // ===============================================================================
        // If current item is a directory, recursively process it
        // This ensures all subfolders are scanned for fluorescence images
        if (File.isDirectory(dir + list[i])) {
            processFolder(dir + list[i]); // Recursive call for subdirectories
        }
        
        // ===============================================================================
        // FLUORESCENCE IMAGE DETECTION AND PROCESSING
        // ===============================================================================
        // Check if current file is a fluorescence image from target channels
        // Only process files ending with specific channel markers
        else if (endsWith(list[i], "_N2A_ORG.tif") || 
                 endsWith(list[i], "_N2B_ORG.tif") || 
                 endsWith(list[i], "_PSD_ORG.tif")) {

            // ===============================================================================
            // IMAGE LOADING AND PREPARATION
            // ===============================================================================
            // Open the original fluorescence image
            open(dir + list[i]);
            origTitle = getTitle(); // Store original image title for reference

            // ===============================================================================
            // BIT DEPTH CONVERSION
            // ===============================================================================
            // Convert to 16-bit for enhanced dynamic range
            // This preserves subtle fluorescence variations and improves quantitative analysis
            run("16-bit");

            // ===============================================================================
            // BACKGROUND ESTIMATION PIPELINE
            // ===============================================================================
            // Step 1: Create duplicate for background processing
            // This preserves the original image while creating a copy for blurring
            run("Duplicate...", "title=blurred");
            
            // Step 2: Apply large Gaussian blur for background estimation
            // Sigma=80 creates a very smooth background approximation
            // This effectively captures uneven illumination patterns
            run("Gaussian Blur...", "sigma=80");

            // ===============================================================================
            // BACKGROUND SUBTRACTION
            // ===============================================================================
            // Subtract the blurred background FROM the original image
            // This removes uneven illumination while preserving fluorescence signal
            // Formula: Cleaned = Original - Background
            imageCalculator("Subtract create", origTitle, "blurred");

            // ===============================================================================
            // RESULT IMAGE HANDLING
            // ===============================================================================
            // Select the result image (last opened window)
            // This is the cleaned fluorescence image after background subtraction
            selectImage(nImages);

            // ===============================================================================
            // FILE NAMING AND SAVING
            // ===============================================================================
            // Create standardized filename for cleaned image
            // Preserves original "_ORG" suffix and adds "_cleaned" identifier
            cleanedName = replace(list[i], ".tif", "_cleaned.tif");
            rename(cleanedName); // Update image window title

            // Save cleaned image in the same subfolder as original
            // Maintains folder structure and organization
            saveAs("Tiff", dir + cleanedName);

            // ===============================================================================
            // MEMORY MANAGEMENT AND CLEANUP
            // ===============================================================================
            // Close all image windows to free memory
            // Essential for processing large batches of images
            close("*");
        }
    }
}

// ===============================================================================
// PROCESSING COMPLETION
// ===============================================================================
//
// The macro automatically completes when all subfolders have been processed
// No additional user intervention required after initial folder selection
//
// Quality Control Notes:
// - All original files remain unchanged
// - Cleaned images are saved with "_cleaned" suffix
// - 16-bit format preserves quantitative accuracy
// - Background subtraction improves signal-to-noise ratio
//
// ===============================================================================
