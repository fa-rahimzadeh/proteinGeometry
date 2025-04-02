import numpy as np
import os

def load_protein_data(protein_id, data_dir):
    """
    Load geometric data for a given protein ID from .npz files.

    Parameters:
    - protein_id (str): The Protein ID to search for.
    - data_dir (str): Directory containing the .npz geometry files.

    Returns:
    - dict: A dictionary containing the protein's geometry data.
    """
    file_path = os.path.join(data_dir, f"{protein_id}_geometry.npz")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for Protein ID: {protein_id}")

    # Load data from the .npz file
    data = np.load(file_path)

    # Extract components
    protein_data = {
        "dist_matrix": data["dist_matrix"],
        "phi_vector": data["phi_vector"],
        "psi_vector": data["psi_vector"],
        "omega_vector": data["omega_vector"],
        "start_index": data["start_index"],
        "end_index": data["end_index"],
        "resolution_classification": data["resolution_classification"],
        "mask": data["mask"],
    }

    return protein_data