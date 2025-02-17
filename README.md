# proteinGeometry
Here's an example of what you can include in the **README** file for your repository. It should provide essential information for users about the dataset, how to use it, and any relevant details for downloading, citing, or contributing.

---

# **Comprehensive Geometric Feature Dataset for Protein Structures**

## **Overview**

This repository contains a comprehensive dataset of geometric features calculated for 6,476 single-chain protein sequences sourced from the SCOPe 2.08 database. The dataset provides crucial structural information, including **pairwise Cβ-Cβ distance matrices** and **backbone torsion angles** (ϕ, ψ, ω), for proteins ranging in length from 25 to 700 residues. The dataset is designed to support various applications in **protein structure prediction**, **molecular simulations**, and **structural bioinformatics**. It also aids in **machine learning** and **drug discovery**, reducing preprocessing time for computational studies.

## **Dataset Details**

The dataset includes **two categories** of structures:
1. **High-resolution Structures (6,464 proteins)** – Full atomic data (Cβ atoms for distance maps and dihedral angles ϕ, ψ, ω).
2. **Low-resolution Structures (12 proteins)** – Only Cα atoms available (distance maps and no dihedral angles).

### **File Format**

Each protein’s geometric information is stored in a **compressed `.npz`** file format containing the following data:

- **dist_matrix**: Pairwise distance matrix representing Cβ-Cβ (Cα for glycine) distances.
- **phi_vector**: Backbone torsion angles (ϕ) or `None` for low-resolution structures.
- **psi_vector**: Backbone torsion angles (ψ) or `None`.
- **omega_vector**: Backbone torsion angles (ω) or `None`.
- **start_index**: The index where the protein sequence begins.
- **end_index**: The index where the protein sequence ends.
- **resolution_classification**: Label indicating whether the protein is high-resolution or low-resolution.
- **mask**: Binary mask indicating missing residues.

### **Data Description**

- **Protein Length Range**: 25 to 700 residues.
- **Data Coverage**: Includes both high-resolution (6,464 proteins) and low-resolution (12 proteins) structures.
- **Resolution**: High-resolution structures contain full atomic data, while low-resolution structures include only Cα atoms and distance maps.

## **Applications**

This dataset can be used for:
- **Protein Structure Prediction**: By leveraging distance maps and dihedral angles, you can predict protein conformations.
- **Molecular Dynamics Simulations**: Provides geometric constraints for better conformational sampling.
- **Structural Bioinformatics**: Facilitates analysis of protein folding, structural homology, and protein-ligand interactions.
- **Machine Learning & AI**: Reduces preprocessing time for neural network training.
- **Drug Discovery**: Used in ligand binding and docking studies to improve drug design.

## **How to Use**

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   ```
2. Ensure you have the required dependencies installed:
   ```bash
   pip install numpy
   ```

### **Loading the Data**

You can load the data using `numpy` to read the `.npz` files. Example code:
```python
import numpy as np

# Load a protein's data
data = np.load('path/to/protein_file.npz')

# Access individual components
dist_matrix = data['dist_matrix']
phi_vector = data['phi_vector']
psi_vector = data['psi_vector']
omega_vector = data['omega_vector']
mask = data['mask']
```

## **Citation**

If you use this dataset in your research, please cite the following reference:

> [Your Citation Here]

## **Contributing**

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## **License**

This dataset is made available under the [MIT License](LICENSE). Feel free to use and distribute the data for academic and research purposes.

---

This template covers the following:
- **Overview** of the dataset and its applications
- **Details** about the file format and what the files contain
- **How to use** the data, including installation and code for loading the files
- **Citation** and licensing information for proper attribution
- **Contributing** instructions for those who want to contribute

Feel free to adapt or expand it as needed!
