# Machine Learning - Assignment 3

## Files present  

- `src` directory:  
    - `requirements.txt` - Contains all the necessary dependencies with their versions.
    - `data_processing.py` - Contains necessary functions to process the data.
    - `feature_extraction.py` - Contains the funstions for principal component analysis and linear discriminant analysis.
    - `svm.py` - Contains the code to choose the best SVM.
    - `metrics.py` - Contains the functions to evaluate the performance of our classification.
    - `main.py` - Main file for completing all tasks required.

- `occupancy_data` directory:
    - `datatraining.txt` - Dataset file 1.
    - `datatest.txt` - Dataset file 2.
    - `datatest_2.txt` - Dataset file 3.

- `plots` directory:
    - `pca.png` - Reduced 2-D data for the train split after PCA.
    - `lda.png` - Reduced 1-D data for the train split after LDA.
    - `pca_scree.png` - Scree plot for PCA.

- `output.txt` - The output for a run of the entire code.

- `ML_Assn_3_Report.pdf` - A report containing the step-wise description of the implementation and analysis of results.

- `README.md` - A README file describing the files present and the instructions to execute the code.


## Instructions to execute the code

- Navigate to the `src` directory.
- Ensure you are using a latest version of Python3, and install all dependencies.  
`pip install -r requirements.txt`
- Execute the file `main.py`  
`python main.py`
- The relevant output will be displayed on the terminal or console.
- The plots will be created and saved in the `plots` directory.