# Protein Geometry Dataset Toolkit  

This repository contains Python scripts for computing and analyzing geometric features of protein structures, including distance maps and dihedral angles. These tools facilitate structural bioinformatics research by providing transformation-invariant representations of protein conformations.

## Contents  

- **`geometryCalc.py`** – Computes dihedral angles (ϕ, ψ, ω) and distance maps from protein structures.  
- **`RamachandranPlot.py`** – Generates Ramachandran plots to visualize backbone torsion angles.  
- **`ReadProteinData.py`** – Loads and retrieves precomputed geometry data for a given Protein ID.

## Installation  

To run these scripts, ensure you have Python installed along with the required dependencies:  

```bash
pip install numpy matplotlib pdbecif numba psutil
```  
## Dataset

The dataset is available on Zenodo: [https://zenodo.org/uploads/14880546]

Each .npz file is named as {protein_id}_geometry.npz, where {protein_id} is the corresponding PDB or SCOPe identifier. Each file contains:

dist_matrix – Pairwise residue Cβ-Cβ (or Cα for Glycine) distances.

phi_vector, psi_vector, omega_vector – Backbone dihedral angles (ϕ, ψ, ω).

start_index, end_index – Indices defining structured regions.

resolution_classification – Classification based on structure resolution.

mask – A mask indicating missing or unreliable data.
## Usage  

### 1. Compute Dihedral Angles and Distance Maps  
Run `geometryCalc.py` with a protein structure file (e.g., PDB format):  

```bash
python geometryCalc.py --input_dir path/to/cif_files --output_dir path/to/output
```  

### 2. Generate a Ramachandran Plot  
Use `RamachandranPlot.py` to visualize backbone torsion angles:  

```bash-
python RamachandranPlot.py --data path/to/generated_geometry_files
```  

### 3. Reading Protein Geometry Data
Use ReadProteinData.py to load and inspect a protein's geometry according to its related protein_id

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
