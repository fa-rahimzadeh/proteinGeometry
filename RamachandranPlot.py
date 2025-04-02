import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm  # Progress bar for large dataset processing

# Define folder containing .npz files
dataset_path = "path_to_geometry_files_directory"

# Lists to store valid phi and psi angles
phi_all, psi_all = [], []

# Process each file
for file in tqdm(os.listdir(dataset_path)):  # Loop over all .npz files
    if file.endswith("_geometry.npz"):  # Ensure we only process geometry files
        data = np.load(os.path.join(dataset_path, file), allow_pickle=True)

        # Check resolution classification (process only high-resolution structures)
        if data["resolution_classification"] == "low_resolution":  
            continue  # Skip low-resolution structures

        # Extract dihedral angles and mask
        phi_vector = data["phi_vector"]  # Phi (ϕ) angles
        psi_vector = data["psi_vector"]  # Psi (ψ) angles
        mask = data["mask"]  # Boolean mask (True = valid, False = invalid)
        
        false_indices = np.where(mask == False)[0]  # Indices where mask is False
        # Copy the mask to modify it safely
        new_mask = mask.copy()
        
        # Ensure first and last residues are always masked
        new_mask[0] = False  
        new_mask[-1] = False  

        # Mask one residue before each False range
        before_false_indices = false_indices - 1
        before_false_indices = before_false_indices[(before_false_indices >= 0)]  # Avoid negative indices
        new_mask[before_false_indices] = False  
        
        # Mask one residue after each False range
        after_false_indices = false_indices + 1
        after_false_indices = after_false_indices[(after_false_indices < len(mask))]  # Avoid out-of-bounds indices
        new_mask[after_false_indices] = False  
        
        # Apply mask to filter valid angles
        valid_indices = new_mask  # Updated mask with additional exclusions
        phi_filtered = phi_vector[valid_indices]
        psi_filtered = psi_vector[valid_indices]
        nan_count = np.isnan(phi_filtered).sum()  # Count NaN values

        # Append valid angles to the lists
        phi_all.extend(phi_filtered)
        psi_all.extend(psi_filtered)

# Convert to NumPy arrays and change radians to degrees
phi_all = np.degrees(np.array(phi_all))
psi_all = np.degrees(np.array(psi_all))

# Ensure phi_all and psi_all are truly 1D
phi_all = phi_all.reshape(-1)  # Converts (N,1) -> (N,)
psi_all = psi_all.reshape(-1)  # Converts (N,1) -> (N,)
print(phi_all.shape)
print(f"Phi range: {min(phi_all)} to {max(phi_all)}")
print(f"Psi range: {min(psi_all)} to {max(psi_all)}")

# Plot Ramachandran plot
plt.figure(figsize=(16, 12))
sns.kdeplot(x=phi_all, y=psi_all, cmap="Blues", fill=True, levels=30, alpha=0.7)
plt.scatter(phi_all, psi_all, s=1, color="purple", alpha=0.3)

# Expected α-Helix and β-Sheet regions
plt.axhline(y=-45, color="r", linestyle="--", label="α-Helix (expected)")
plt.axhline(y=135, color="g", linestyle="--", label="β-Sheet (expected)")

# Plot settings
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.xlabel("Phi (ϕ) Angle [°]")
plt.ylabel("Psi (ψ) Angle [°]")
plt.title("Ramachandran Plot of High-Resolution Protein Structures")
plt.legend()
plt.grid(True)
plt.show()
print("Ramachandran Plot is Ready")