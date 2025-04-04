# Protein Geometry Dataset Toolkit  

This repository contains Python scripts for computing and analyzing geometric features of protein structures, including distance maps and dihedral angles. These tools facilitate structural bioinformatics research by providing transformation-invariant representations of protein conformations.

## Contents  

- **`geometryCalc.py`** – Computes dihedral angles (ϕ, ψ, ω) (1D) and distance maps (2D) from protein structures.  
- **`RamachandranPlot.py`** – Generates Ramachandran plots to visualize backbone torsion angles.  
- **`ReadProteinData.py`** – Loads and retrieves precomputed geometry data for a given Protein ID.

## Installation  

To run these scripts, ensure you have Python installed along with the required dependencies:  

```bash
pip install numpy matplotlib pdbecif numba psutil
```  

## Usage  

### 1. Compute Dihedral Angles and Distance Maps  
First define the Input and Output Directories

- #### Input Directory: _ Contains **`.cif`** files

- #### Output Directory: _ Stores **`.npz`** geometry files

Modify these paths in **`geometryCalc.py`**:

```bash
input_directory = 'path/to/input_cifs/'  
output_directory = 'path/to/output_geometry/'
```

Then, use the following command to process all **`.cif`** files in the input directory:

```bash
python geometryCalc.py 
```  

### 2. Generate a Ramachandran Plot  
First define the Input Geometry Files Directory
Input directory: Contains**`.npz`** geometry files

Then, use `RamachandranPlot.py` to visualize backbone torsion angles:  

```bash-
python RamachandranPlot.py 
```  

### 3. Reading Protein Geometry Data
Use ReadProteinData.py to load and inspect a protein's geometry according to its related protein_id as bellow


```bash
from ReadProteinData import load_protein_data

# Define data directory
data_directory = "path/to/geometry_files"

# Load data for a specific protein ID
protein_id = "7bvt"
protein_data = load_protein_data(protein_id, data_directory)

# Print basic details
print(f"Loaded {protein_id} successfully.")
print(f"Distance Matrix Shape: {protein_data['dist_matrix'].shape}")
```

## License  


## Citation  

---
