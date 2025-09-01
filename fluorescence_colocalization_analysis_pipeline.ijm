// ===============================================================================
// FLUORESCENCE COLOCALIZATION ANALYSIS PIPELINE
// ===============================================================================
//
// Author: Sailaja Kuruvada
// Date: 2025
// Purpose: Automated colocalization analysis of synaptic protein markers in fluorescence microscopy
//
// This ImageJ macro performs batch colocalization analysis of fluorescence microscopy images to:
// 1. Analyze spatial overlap between synaptic protein markers (N2A, N2B, PSD95)
// 2. Process only 60X magnification images for high-resolution analysis
// 3. Integrate with JACoP (Just Another Colocalization Plugin) for statistical analysis
// 4. Generate comprehensive colocalization metrics and correlation coefficients
// 5. Compile all results into a single master log file for statistical analysis
//
// INPUT REQUIREMENTS:
// - Main folder: "IMAGE ANALYSIS NEW" containing subfolders with cleaned fluorescence images
// - File naming convention: "*_N2A_ORG_cleaned.tif", "*_N2B_ORG_cleaned.tif", "*_PSD_ORG_cleaned.tif"
// - Image format: Background-subtracted TIFF files from cleaning pipeline
// - Magnification filter: Only 60X images for high-resolution analysis
//
// OUTPUT RESULTS:
// - Master log file: "coloc_master_log.txt" containing all colocalization results
// - JACoP analysis results: Pearson correlation, Manders coefficients, Costes analysis
// - Colocalization metrics: Spatial overlap measurements between synaptic markers
// - Statistical data: Ready for import into statistical analysis software
//
// ANALYSIS PAIRS:
// - N2A vs PSD95: NR2A subunit colocalization with postsynaptic density
// - N2B vs PSD95: NR2B subunit colocalization with postsynaptic density
// - Focus on 60X magnification for optimal spatial resolution
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
// - indexOf(): Magnification filtering (60X only)
//
// Image Processing Operations:
// - open(): Image loading and display for colocalization analysis
// - close(): Memory management after analysis completion
//
// User Interaction Operations:
// - waitForUser(): Pause for manual JACoP plugin execution
// - print(): Progress reporting and result logging
//
// File Management Operations:
// - File.saveString(): Master log file export
// - getInfo("log"): Current log window content retrieval
//
// ===============================================================================
// JACoP PLUGIN INTEGRATION
// ===============================================================================
//
// JACoP (Just Another Colocalization Plugin) Requirements:
// - Must be installed in ImageJ/Fiji
// - Provides statistical colocalization analysis
// - Calculates Pearson correlation coefficients
// - Computes Manders overlap coefficients
// - Performs Costes randomization test
// - Generates scatter plots and colocalization maps
//
// Manual Execution Required:
// - User must run JACoP plugin manually for each image pair
// - Ensures proper parameter selection and validation
// - Allows for quality control during analysis
//
// ===============================================================================

// Main execution block - initiates the colocalization analysis pipeline
mainDir = getDirectory("Choose the main folder (IMAGE ANALYSIS NEW)");
logFile = mainDir + "coloc_master_log.txt";

// ===============================================================================
// LOG INITIALIZATION AND HEADER
// ===============================================================================
// Clear previous log entries and create new analysis session
print("\\Clear"); 
print(">>> Starting colocalization batch (60X only)");
print("Results will be saved in: " + logFile);
print("=================================================");

// Execute the main processing pipeline
processFolder(mainDir);

// ===============================================================================
// FINAL LOG EXPORT
// ===============================================================================
// Save all accumulated results to master log file
print(">>> All done! Saving master log to: " + logFile);
File.saveString(getInfo("log"), logFile);

// ===============================================================================
// CORE PROCESSING FUNCTION
// ===============================================================================
//
// Function: processFolder(dir)
// Purpose: Recursively processes all subfolders and performs colocalization analysis
//
// Parameters:
// - dir: String path to current directory being processed
//
// Processing Logic:
// 1. Directory scanning: Lists all files and subdirectories
// 2. Recursive processing: Calls itself for subdirectories
// 3. File filtering: Identifies cleaned fluorescence images (60X only)
// 4. Image pairing: Matches N2A/N2B with PSD95 for colocalization analysis
// 5. JACoP integration: Facilitates manual plugin execution
// 6. Result logging: Captures all analysis outputs
//
// Channel Pairing Strategy:
// - N2A vs PSD95: Analyzes NR2A subunit colocalization with postsynaptic density
// - N2B vs PSD95: Analyzes NR2B subunit colocalization with postsynaptic density
// - PSD95 serves as reference marker for synaptic localization
//
// ===============================================================================

function processFolder(dir) {
    // Get list of all files and directories in current folder
    list = getFileList(dir);
    
    // Initialize file variables for each channel type
    n2aFile = ""; n2bFile = ""; psdFile = "";

    // ===============================================================================
    // FILE DISCOVERY AND FILTERING
    // ===============================================================================
    // Iterate through each item in the directory
    for (i=0; i<list.length; i++) {
        
        // ===============================================================================
        // RECURSIVE DIRECTORY PROCESSING
        // ===============================================================================
        // If current item is a directory, recursively process it
        // This ensures all subfolders are scanned for fluorescence images
        if (File.isDirectory(dir+list[i])) {
            processFolder(dir+list[i]); // Recursive call for subdirectories
        }
        
        // ===============================================================================
        // 60X FLUORESCENCE IMAGE DETECTION
        // ===============================================================================
        // Check for N2A channel images (60X magnification only)
        // Filters for cleaned images from the background subtraction pipeline
        else if (endsWith(list[i], "_N2A_ORG_cleaned.tif") && indexOf(list[i], "60X") >= 0) {
            n2aFile = list[i]; // Store N2A file for colocalization analysis
        }
        
        // Check for N2B channel images (60X magnification only)
        // Filters for cleaned images from the background subtraction pipeline
        else if (endsWith(list[i], "_N2B_ORG_cleaned.tif") && indexOf(list[i], "60X") >= 0) {
            n2bFile = list[i]; // Store N2B file for colocalization analysis
        }
        
        // Check for PSD95 channel images (60X magnification only)
        // Filters for cleaned images from the background subtraction pipeline
        else if (endsWith(list[i], "_PSD_ORG_cleaned.tif") && indexOf(list[i], "60X") >= 0) {
            psdFile = list[i]; // Store PSD95 file for colocalization analysis
        }
    }

    // ===============================================================================
    // N2A vs PSD95 COLOCALIZATION ANALYSIS
    // ===============================================================================
    // Perform colocalization analysis between NR2A and PSD95 markers
    // This analyzes the spatial relationship between NMDA receptor subunits and postsynaptic density
    if (n2aFile != "" && psdFile != "") {
        
        // Load both images for colocalization analysis
        open(dir + n2aFile); // Open N2A channel image
        open(dir + psdFile); // Open PSD95 channel image
        
        // Log the current analysis pair
        print("\nProcessing: " + n2aFile + " vs " + psdFile);
        
        // ===============================================================================
        // JACoP PLUGIN INTEGRATION
        // ===============================================================================
        // Pause for manual JACoP plugin execution
        // User must run JACoP manually to ensure proper parameter selection
        // This allows for quality control and validation during analysis
        waitForUser("Run JACoP manually on:\n" + n2aFile + "\nvs\n" + psdFile + "\n\nPress OK after JACoP finishes.");
        
        // Clean up memory after analysis completion
        close("*");
    }

    // ===============================================================================
    // N2B vs PSD95 COLOCALIZATION ANALYSIS
    // ===============================================================================
    // Perform colocalization analysis between NR2B and PSD95 markers
    // This analyzes the spatial relationship between different NMDA receptor subunits and postsynaptic density
    if (n2bFile != "" && psdFile != "") {
        
        // Load both images for colocalization analysis
        open(dir + n2bFile); // Open N2B channel image
        open(dir + psdFile); // Open PSD95 channel image
        
        // Log the current analysis pair
        print("\nProcessing: " + n2bFile + " vs " + psdFile);
        
        // ===============================================================================
        // JACoP PLUGIN INTEGRATION
        // ===============================================================================
        // Pause for manual JACoP plugin execution
        // User must run JACoP manually to ensure proper parameter selection
        // This allows for quality control and validation during analysis
        waitForUser("Run JACoP manually on:\n" + n2bFile + "\nvs\n" + psdFile + "\n\nPress OK after JACoP finishes.");
        
        // Clean up memory after analysis completion
        close("*");
    }
}

// ===============================================================================
// ANALYSIS COMPLETION
// ===============================================================================
//
// The macro automatically completes when all subfolders have been processed
// All colocalization results are compiled into a single master log file
//
// Quality Control Notes:
// - Only 60X magnification images are processed for optimal resolution
// - Manual JACoP execution ensures proper parameter validation
// - All results are logged and saved for statistical analysis
// - Memory management prevents system overload during batch processing
//
// Expected JACoP Outputs:
// - Pearson correlation coefficient (linear relationship)
// - Manders overlap coefficients (M1, M2 for each channel)
// - Costes randomization test (significance testing)
// - Scatter plots and colocalization maps
// - Threshold optimization results
//
// ===============================================================================
