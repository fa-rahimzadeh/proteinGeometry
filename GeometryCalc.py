import os
import numpy as np
import logging
import gc
import psutil
from glob import glob
from pdbecif.mmcif_tools import MMCIF2Dict
from numba import njit
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("protein_geometry_processing.log", mode='a')
    ]
)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    logging.info(f"Current memory usage: {mem:.2f} MB")

@njit
def calculate_dihedral(p1, p2, p3, p4, epsilon=1e-6):
    """
    Calculate the dihedral (torsion) angle between four 3D points.
    Returns angle in degrees.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normalize b2 (avoid division by zero)
    b2_norm = np.linalg.norm(b2)
    if b2_norm < epsilon:
        return 0.0  # Prevent division by zero
    
    b2 /= b2_norm

    # Compute normal vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize normal vectors
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)

    if norm_n1 < epsilon or norm_n2 < epsilon:
        return 0.0  # Avoid division by zero

    n1 /= norm_n1
    n2 /= norm_n2

    # Compute m vector
    m1 = np.cross(n1, b2)

    # Compute torsion angle
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return -np.arctan2(y, x)

class DihedralCache:
    def __init__(self):
        self.cache = {}
    def get_or_compute(self, p1, p2, p3, p4):
        key = (tuple(p1), tuple(p2), tuple(p3), tuple(p4))
        if key in self.cache:
            return self.cache[key]
        angle = calculate_dihedral(p1, p2, p3, p4)
        self.cache[key] = angle
        return angle

def calculate_distance_matrix(coords):
    return np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)

def interpolate_coordinates(p1, p2, num_points):
    # Ensure p1 and p2 are numpy arrays
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return [p1 + (p2 - p1) * (i + 1) / (num_points + 1) for i in range(num_points)]

def is_none_or_nan_array(value):
    """Helper function to check if a value is None or a numpy array with all elements as NaN."""
    return value is None or (isinstance(value, np.ndarray) and np.isnan(value).all())

def create_atom_dicts(seq_ids, atom_site, CA_coords, N_coords, C_coords, CB_coords):
    """
    Creates dictionaries for CA, N, C, and CB atoms for each seq_id based on specified conditions.
    If an atom type is missing for a seq_id, assigns None.
    Special rules apply for GLY and UNK residues regarding CB atoms.
    """
    # Initialize dictionaries for each atom type
    ca_dict = {}
    n_dict = {}
    c_dict = {}
    cb_dict = {}
    
    
    
    # Index to access coordinates in CA, N, C, CB lists
    ca_idx, n_idx, c_idx, cb_idx = 0, 0, 0, 0

    for seq_id in seq_ids:
               
        # Convert `sid` to int for comparison with `seq_id` and filter indices
        try:
            related_indices = [
                i for i, sid in enumerate(atom_site['label_seq_id']) if sid.isdigit() and int(sid) == seq_id
                ]
        except ValueError as e:
            logging.error(f"Conversion error with `label_seq_id` in atom_site: {e}")
            continue

        # Ensure `related_indices` contains valid indices
        if not related_indices:
            logging.warning(f"No related indices found for seq_id {seq_id}")
            continue

        # Extract `label_atom_id` and `label_comp_id` using valid integer indices
        try:
            related_atom_ids = [atom_site['label_atom_id'][int(i)] for i in related_indices]
            related_comp_id = atom_site['label_comp_id'][int(related_indices[0])] if related_indices else None
        except (IndexError, ValueError, TypeError) as e:
            logging.error(f"Error accessing `label_atom_id` or `label_comp_id` in atom_site: {e}")
            continue

        # CA Atom
        if 'CA' in related_atom_ids:
            ca_dict[seq_id] = CA_coords[ca_idx] if ca_idx < len(CA_coords) else None
            ca_idx += 1
        else:
            ca_dict[seq_id] = None

        # N Atom
        if 'N' in related_atom_ids:
            n_dict[seq_id] = N_coords[n_idx] if n_idx < len(N_coords) else None
            n_idx += 1
        else:
            n_dict[seq_id] = None

        # C Atom
        if 'C' in related_atom_ids:
            c_dict[seq_id] = C_coords[c_idx] if c_idx < len(C_coords) else None
            c_idx += 1
        else:
            c_dict[seq_id] = None

        # CB Atom or special handling for GLY and UNK
        if related_comp_id in ['GLY', 'UNK']:
            if 'CA' in related_atom_ids:
                cb_dict[seq_id] = CB_coords[cb_idx] if cb_idx < len(CB_coords) else None
                cb_idx += 1
            else:
                cb_dict[seq_id] = None
        else:
            if 'CB' in related_atom_ids:
                cb_dict[seq_id] = CB_coords[cb_idx] if cb_idx < len(CB_coords) else None
                cb_idx += 1
            else:
                cb_dict[seq_id] = None
    
    return ca_dict, n_dict, c_dict, cb_dict

def infer_missing_atoms(seq_ids, atom_site, CA_coords, N_coords, C_coords, CB_coords):
    """discontinous
    Infers or interpolates missing atoms in each residue based on neighboring atoms.
    Uses create_atom_dicts to initialize dictionaries for CA, N, C, and CB atoms.
    """
    # Create initial atom dictionaries with existing or None values
    ca_dict, n_dict, c_dict, cb_dict = create_atom_dicts(seq_ids, atom_site, CA_coords, N_coords, C_coords, CB_coords)

    inferred_coords = {'CA': [], 'N': [], 'C': [], 'CB': []}
    valid_seq_ids = []
    
    for idx, seq_id in enumerate(seq_ids):
        # Retrieve current residue coordinates or None if missing
        ca_coord = ca_dict.get(seq_id)
        n_coord = n_dict.get(seq_id)
        c_coord = c_dict.get(seq_id)
        cb_coord = cb_dict.get(seq_id)

        # Check continuity: ensure that missing atoms in a residue mean that the residue is contiguous
        if any(is_none_or_nan_array(coord) for coord in (ca_coord, n_coord, c_coord, cb_coord)):
            is_discontinuous = (
                (idx == 0 or seq_ids[idx - 1] != seq_id - 1) or
                (idx == len(seq_ids) - 1 or seq_ids[idx + 1] != seq_id + 1)
            )
            if is_discontinuous:
                continue  # Skip non-continuous seq_id
              
        # Interpolate missing CA atom using neighboring residues
        if ca_coord is None:
            if idx > 0 and idx < len(seq_ids) - 1:
                prev_ca = ca_dict.get(seq_ids[idx - 1])
                next_ca = ca_dict.get(seq_ids[idx + 1])
                if prev_ca is not None and next_ca is not None:
                    ca_coord = (prev_ca + next_ca) / 2
        
        # Interpolate missing N atom using previous C and CA atoms
        if n_coord is None:
            if idx > 0:
                prev_c = c_dict.get(seq_ids[idx - 1])
                if prev_c is not None and ca_coord is not None:
                    n_coord = (prev_c + ca_coord) / 2
        
        # Interpolate missing C atom using CA and next N atoms
        if c_coord is None:
            if idx < len(seq_ids) - 1:
                next_n = n_dict.get(seq_ids[idx + 1])
                if ca_coord is not None and next_n is not None:
                    c_coord = (ca_coord + next_n) / 2
        
        # Interpolate missing CB atom based on CA; handle for GLY and UNK if necessary
        if cb_coord is None:
            # Use integer indexing to get the component ID for the current `seq_id`
            related_indices = [
                i for i, sid in enumerate(atom_site['label_seq_id']) if sid.isdigit() and int(sid) == seq_id
                ]
            if related_indices:
                related_comp_id = atom_site['label_comp_id'][related_indices[0]]  # Get comp_id as integer-indexed
                if related_comp_id in ['GLY', 'UNK']:
                    cb_coord = ca_coord if ca_coord is not None else None  # Handle missing CA case
                else:
                    if ca_coord is not None:
                        cb_coord = ca_coord + np.array([0.5, 0.5, 0.5])  # Simplified placeholder for CB position
        
        # Append inferred or original coordinates to the results
        inferred_coords['CA'].append(ca_coord)
        inferred_coords['N'].append(n_coord)
        inferred_coords['C'].append(c_coord)
        inferred_coords['CB'].append(cb_coord)
        valid_seq_ids.append(seq_id)
        
    # Convert lists to numpy arrays, using dtype=object to handle potential Nones
    inferred_coords['CA'] = np.array(inferred_coords['CA'], dtype=object)
    inferred_coords['N'] = np.array(inferred_coords['N'], dtype=object)
    inferred_coords['C'] = np.array(inferred_coords['C'], dtype=object)
    inferred_coords['CB'] = np.array(inferred_coords['CB'], dtype=object)

    return valid_seq_ids, inferred_coords
    
def handle_small_gaps(sequence_indices, coordinates):
    filled_coordinates = []
    if len(sequence_indices) != len(coordinates):
        logging.error(
            f"Inconsistent lengths: sequence_indices has {len(sequence_indices)} items, "
            f"while coordinates has {len(coordinates)} items."
        )
        return filled_coordinates  # Return an empty list if there's a mismatch
    try:
        for i in range(len(sequence_indices) - 1):
            current_index, next_index = sequence_indices[i], sequence_indices[i + 1]
            current_coord, next_coord = coordinates[i], coordinates[i + 1]
            filled_coordinates.append(current_coord)
            
            gap_size = next_index - current_index - 1
            if 1 <= gap_size <= 2:
                interpolated_coords = interpolate_coordinates(current_coord, next_coord, gap_size)
                filled_coordinates.extend(interpolated_coords)
        
        filled_coordinates.append(coordinates[-1])

    except IndexError as e:
        logging.error(f"Index error in handle_small_gaps at i={i}: {e}")
        
    return filled_coordinates

def handle_large_gaps(sequence_indices, max_gap_size=3):
    """
    Mask coordinates for moderate to large gaps.
    Returns a mask indicating which residues are valid.
    """
    # Create a mask of length that spans the sequence range
    mask = np.ones((sequence_indices[-1] - sequence_indices[0] + 1), dtype=bool)  # Start with all True
    for i in range(len(sequence_indices) - 1):
        current_index, next_index = sequence_indices[i], sequence_indices[i + 1]
        gap_size = next_index - current_index - 1
        if gap_size >= max_gap_size:
            mask[current_index + 1 - sequence_indices[0] : next_index - sequence_indices[0]] = False  # Mask the gap region
    return mask 

def average_atom_positions(atom_indices, atom_site):
    grouped_atoms = defaultdict(list)
    for i in atom_indices:
        seq_id = atom_site['label_seq_id'][i]
        atom_id = atom_site['label_atom_id'][i]
        occupancy = float(atom_site.get('occupancy', [1.0])[i])
        x = float(atom_site['Cartn_x'][i])
        y = float(atom_site['Cartn_y'][i])
        z = float(atom_site['Cartn_z'][i])
        grouped_atoms[(seq_id, atom_id)].append((x, y, z, occupancy))
    averaged_cartn = []
    for group, coords in grouped_atoms.items():
        if len(coords) == 1:
            averaged_cartn.append(coords[0][:3])
        else:
            total_occ = sum(occ for _, _, _, occ in coords)
            avg_x = sum(x * occ for x, _, _, occ in coords) / total_occ
            avg_y = sum(y * occ for _, y, _, occ in coords) / total_occ
            avg_z = sum(z * occ for _, _, _, occ in coords) / total_occ
            averaged_cartn.append((avg_x, avg_y, avg_z))
    return averaged_cartn

def check_skip_first_last_residues(seq_ids, atom_CA, atom_C, atom_N, atom_CB, atom_site):
    """
    Determines if the first or last residues should be skipped based on the presence of necessary atoms.
    """
    try:
        if not seq_ids:
            return True, True  # If no sequence IDs, skip both
        
        first_residue_idx = seq_ids[0]
        last_residue_idx = seq_ids[-1]

        # Get component IDs for first and last residues
        first_residue_comp_id = atom_site['label_comp_id'][atom_CA[0]] if atom_CA else "UNK"
        last_residue_comp_id = atom_site['label_comp_id'][atom_CA[-1]] if atom_CA else "UNK"

        # Check for required atoms in the first residue
        has_CA_first = (
            atom_site['label_atom_id'][atom_CA[0]] == 'CA' and 
            int(atom_site['label_seq_id'][atom_CA[0]]) == first_residue_idx
        )
        has_C_first = (
            atom_site['label_atom_id'][atom_C[0]] == 'C' and 
            int(atom_site['label_seq_id'][atom_C[0]]) == first_residue_idx
        )
        has_N_first = (
            atom_site['label_atom_id'][atom_N[0]] == 'N' and 
            int(atom_site['label_seq_id'][atom_N[0]]) == first_residue_idx
        )
        has_CB_first = (
            (atom_site['label_atom_id'][atom_CB[0]] == 'CB' or 
             (atom_site['label_atom_id'][atom_CB[0]] == 'CA' and first_residue_comp_id in ['GLY', 'UNK']))
            and int(atom_site['label_seq_id'][atom_CB[0]]) == first_residue_idx
        )

        # Check for required atoms in the last residue
        has_CA_last = (
            atom_site['label_atom_id'][atom_CA[-1]] == 'CA' and 
            int(atom_site['label_seq_id'][atom_CA[-1]]) == last_residue_idx
        )
        has_C_last = (
            atom_site['label_atom_id'][atom_C[-1]] == 'C' and 
            int(atom_site['label_seq_id'][atom_C[-1]]) == last_residue_idx
        )
        has_N_last = (
            atom_site['label_atom_id'][atom_N[-1]] == 'N' and 
            int(atom_site['label_seq_id'][atom_N[-1]]) == last_residue_idx
        )
        has_CB_last = (
            (atom_site['label_atom_id'][atom_CB[-1]] == 'CB' or 
             (atom_site['label_atom_id'][atom_CB[-1]] == 'CA' and last_residue_comp_id in ['GLY', 'UNK']))
            and int(atom_site['label_seq_id'][atom_CB[-1]]) == last_residue_idx
        )

        # Remove first residue if required atoms are missing
        if not (has_CA_first and has_C_first and has_N_first and has_CB_first):
            seq_ids = seq_ids[1:]
            if has_CA_first:
                atom_CA = atom_CA[1:] if len(atom_CA) > 1 else []
            if has_C_first:
                atom_C = atom_C[1:] if len(atom_C) > 1 else []
            if has_N_first:
                atom_N = atom_N[1:] if len(atom_N) > 1 else []
            if has_CB_first:
                atom_CB = atom_CB[1:] if len(atom_CB) > 1 else []
              
        # Remove last residue if required atoms are missing
        if not (has_CA_last and has_C_last and has_N_last and has_CB_last):
            seq_ids = seq_ids[:-1]
            if has_CA_last:
                atom_CA = atom_CA[:-1] if len(atom_CA) > 1 else []
            if has_C_last:
                atom_C = atom_C[:-1] if len(atom_C) > 1 else []
            if has_N_last:
                atom_N = atom_N[:-1] if len(atom_N) > 1 else []
            if has_CB_last:
                atom_CB = atom_CB[:-1] if len(atom_CB) > 1 else []

        return seq_ids, atom_CA, atom_C, atom_N, atom_CB

    except IndexError as e:
        logging.error(f"Index error while checking first/last residues: {e}")
        return  # Skip both residues if there's an error

def process_cif_file(entry): 
    log_memory_usage()
    try:
        mmcif_dict = MMCIF2Dict()
        file_name = os.path.splitext(os.path.basename(entry))[0]
        cif_dict = mmcif_dict.parse(entry)
        pdb_id = list(cif_dict.keys())[0]
        atom_site = cif_dict[pdb_id]['_atom_site']
        
        # Extract atom indices for N, CA, C, and CB
        first_model_mask = [
            atom_site['group_PDB'][i] == 'ATOM' and atom_site['pdbx_PDB_model_num'][i] == '1' and atom_site['label_asym_id'][i] == 'A'
            for i in range(len(atom_site['pdbx_PDB_model_num']))
        ]

        atom_N = [i for i in range(len(atom_site['label_atom_id'])) if atom_site['label_atom_id'][i] == 'N' and first_model_mask[i]]
        atom_CA = [i for i in range(len(atom_site['label_atom_id'])) if atom_site['label_atom_id'][i] == 'CA' and first_model_mask[i]]
        atom_C = [i for i in range(len(atom_site['label_atom_id'])) if atom_site['label_atom_id'][i] == 'C' and first_model_mask[i]]
        atom_CB = [
            i for i in range(len(atom_site['label_atom_id'])) 
            if (atom_site['label_atom_id'][i] == 'CB'
                or (atom_site['label_comp_id'][i] == 'GLY' and atom_site['label_atom_id'][i] == 'CA')
                or (atom_site['label_comp_id'][i] == 'UNK' and atom_site['label_atom_id'][i] == 'CA')) and first_model_mask[i]
        ]
       
        seq_ids = sorted(set(
            int(atom_site['label_seq_id'][i]) 
            for i in atom_CA + atom_N + atom_C + atom_CB
            if 'label_seq_id' in atom_site and atom_site['label_seq_id'][i]  # Ensure the field exists and is valid
        ))
        
        if not seq_ids:
            logging.warning(f"No sequence IDs found for {file_name}. Skipping file.")
            return


        # Check for missing atoms in the first and last residues and set skip flags
        classification = 'high_resolution' if len(atom_N) > 0.5 * len(atom_CA) else 'low_resolution'
        
        N_cartn = average_atom_positions(atom_N, atom_site)
        CA_cartn = average_atom_positions(atom_CA, atom_site)
        C_cartn = average_atom_positions(atom_C, atom_site)   
        CB_cartn = average_atom_positions(atom_CB, atom_site)

        # Convert to numpy arrays for distance calculations
        N_coords = np.array(N_cartn, dtype=np.float32)
        CA_coords = np.array(CA_cartn, dtype=np.float32)
        C_coords = np.array(C_cartn, dtype=np.float32)
        CB_coords = np.array(CB_cartn, dtype=np.float32)
        
        # Infer or interpolate missing atoms in each residue
        if classification == 'high_resolution': 
            seq_ids, inferred_atoms = infer_missing_atoms(seq_ids, atom_site, CA_coords, N_coords, C_coords, CB_coords)
            CA_coords = inferred_atoms['CA']
            N_coords = inferred_atoms['N']
            C_coords = inferred_atoms['C']
            CB_coords = inferred_atoms['CB']

            # Process small gaps
            N_coords = handle_small_gaps(seq_ids, N_coords)
            CA_coords = handle_small_gaps(seq_ids, CA_coords)
            C_coords = handle_small_gaps(seq_ids, C_coords)
            CB_coords = handle_small_gaps(seq_ids, CB_coords)
        else:
            CA_coords = handle_small_gaps(seq_ids, CA_coords)
 
        start_index = seq_ids[0]
        end_index = seq_ids[-1]

        # Mask large gaps
        mask = handle_large_gaps(seq_ids)

        # Additional masking: identify residues with only CA coordinates
        if classification == 'high_resolution':
            ca_only_indices = []
            for i, seq_id in enumerate(seq_ids):

                # Check if this residue has only CA and no other atoms
                ca_only = (
                    (N_coords[i] is None or not isinstance(N_coords[i], np.ndarray)) and
                    (C_coords[i] is None or not isinstance(C_coords[i], np.ndarray))
                )
                if ca_only:
                    # Update the mask using the correct index in the full range
                    mask[seq_id - seq_ids[0]] = False
                    ca_only_indices.append(i)

            # Remove the `ca_only` residues from seq_ids and all coordinates
            seq_ids = [seq_id for i, seq_id in enumerate(seq_ids) if i not in ca_only_indices]
            CA_coords = [coord for i, coord in enumerate(CA_coords) if i not in ca_only_indices]
            N_coords = [coord for i, coord in enumerate(N_coords) if i not in ca_only_indices]
            C_coords = [coord for i, coord in enumerate(C_coords) if i not in ca_only_indices]
            CB_coords = [coord for i, coord in enumerate(CB_coords) if i not in ca_only_indices]

        # Prepare coordinates with NaN placeholders for masked (invalid) positions
        coords_for_distance_full = np.full((len(mask), 3), np.nan, dtype=np.float32)  # Assuming 3D coordinates

        # Check structure classification and select appropriate coordinates
        if classification == 'high_resolution':
            # Use CB coordinates for high-resolution structures
            valid_coords = np.array(CB_coords, dtype=np.float32)
        else:
            # Use CA coordinates for low-resolution structures
            valid_coords = np.array(CA_coords, dtype=np.float32)
        
        # Filter valid coordinates based on the mask and place them in the corresponding positions
        coords_for_distance_full[mask] = valid_coords
        print("Shape of coords_for_distance_full:", coords_for_distance_full.shape)
        
        # Calculate distance matrix only on the valid coordinates (ignoring NaNs)
        valid_indices = np.where(mask)[0]  # Indices of valid rows
        dist_matrix_valid = calculate_distance_matrix(coords_for_distance_full[mask])
        
        # Initialize a full distance matrix with NaNs to preserve shape
        dist_matrix_full = np.full((len(mask), len(mask)), np.nan, dtype=np.float32)
        
        # Insert calculated distances into the full distance matrix at valid indices
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                dist_matrix_full[idx_i, idx_j] = dist_matrix_valid[i, j]
        
        print("Shape of dist_matrix_full:", dist_matrix_full.shape)
        
        # Initialize dihedral angles for high and low resolution
        if classification == 'high_resolution' and atom_N and atom_C and atom_CA:
            # Initialize vectors with NaNs for all positions, matching the length of CA_coords
            phi_vector = np.full((len(mask),1), np.nan, dtype=np.float32)
            psi_vector = np.full((len(mask),1), np.nan, dtype=np.float32)
            omega_vector = np.full((len(mask),1), np.nan, dtype=np.float32)
            valid_CA_coords = np.array(CA_coords, dtype=np.float32)
            valid_C_coords = np.array(C_coords, dtype=np.float32)
            valid_N_coords = np.array(N_coords, dtype=np.float32)
            full_CA_coords = np.full((len(mask), 3), np.nan, dtype=np.float32)
            full_C_coords = np.full((len(mask), 3), np.nan, dtype=np.float32)
            full_N_coords = np.full((len(mask), 3), np.nan, dtype=np.float32)
            full_CA_coords[mask] = valid_CA_coords
            full_C_coords[mask] = valid_C_coords
            full_N_coords[mask] = valid_N_coords
            
            n = len(mask)
        
            # Calculate dihedral angles, applying mask
            for i in range(n):
                # Skip calculation if this residue is masked out
                if not mask[i]:
                    continue
        
                # Calculate phi angles
                if i > 0 and mask[i - 1]:  # Ensure previous residue is valid
                    try:
                        p1, p2, p3, p4 = full_C_coords[i - 1], full_N_coords[i], full_CA_coords[i], full_C_coords[i]
                        if all(p is not None for p in (p1, p2, p3, p4)):
                            phi_vector[i] = calculate_dihedral(p1, p2, p3, p4)
                    except IndexError as e:
                        logging.error(f"Index error in phi calculation at index {i}: {e}")
        
                # Calculate psi angles
                if i < n - 1 and mask[i + 1]:  # Ensure next residue is valid
                    try:
                        p1, p2, p3, p4 = full_N_coords[i], full_CA_coords[i], full_C_coords[i], full_N_coords[i + 1]
                        if all(p is not None for p in (p1, p2, p3, p4)):
                            psi_vector[i] = calculate_dihedral(p1, p2, p3, p4)
                    except IndexError as e:
                        logging.error(f"Index error in psi calculation at index {i}: {e}")
        
                # Calculate omega angles
                if i < n - 1 and mask[i + 1]:  # Ensure next residue is valid
                    try:
                        p1, p2, p3, p4 = full_CA_coords[i], full_C_coords[i], full_N_coords[i + 1], full_CA_coords[i + 1]
                        if all(p is not None for p in (p1, p2, p3, p4)):
                            omega_vector[i] = calculate_dihedral(p1, p2, p3, p4)
                    except IndexError as e:
                        logging.error(f"Index error in omega calculation at index {i}: {e}")
        else:
            # Set dihedral angles to None for low-resolution structures
            phi_vector, psi_vector, omega_vector = None, None, None

   	output_directory = 'path_to_output_geometry_directory' 
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(output_dir, f'{file_name}_geometry.npz'),
            dist_matrix=dist_matrix_full,
            phi_vector=phi_vector,
            psi_vector=psi_vector,
            omega_vector=omega_vector,
            start_index=start_index,
            end_index=end_index,
            resolution_classification=classification,
            mask=mask
        )

        log_memory_usage()
        gc.collect()
        
        logging.info(f"Finished processing {file_name}")

    except Exception as e:
        logging.error(f"Error processing file {entry}: {e}")


def main():
    input_directory = 'path_to_input_cif_files_directory'  
    os.makedirs(output_directory, exist_ok=True)

    # Ensure folder_path is indeed a directory
    if not os.path.isdir(input_directory):
        logging.error(f"Provided path is not a directory: {input_directory}")
        return

    # List all `.cif` files in the directory
    cif_files = glob(os.path.join(input_directory, '*.cif'))
    print(len(cif_files))
    if cif_files:
        for file in cif_files:
           file_path = os.path.join(input_directory, file)
           print(file_path)
           if os.path.isfile(file_path):  # Only process files, not directories
               logging.info(f"Processing file: {file_path}")
               # Call the function to process the .cif file
               process_cif_file(file_path)
           else:
               logging.warning(f"Skipped non-file entry: {file_path}")
    else:
       logging.info("No `.cif` files found for processing.")

if __name__ == '__main__':
    main()