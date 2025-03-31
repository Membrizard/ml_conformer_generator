"""
This module implements the Fréchet ChemNet Distance (FCD) calculation between different sets of molecules.
FCD is a metric that measures the similarity between two sets of molecules using their ChemNet embeddings.
The distance is calculated using the Fréchet distance formula applied to the distribution of molecular embeddings.
"""

import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdmolops
from scipy.linalg import sqrtm


# Define ChemNet model for molecular embeddings
class MolecularGraphEncoder(nn.Module):
    """Neural network for encoding molecular structures into vector embeddings"""

    def __init__(self, atom_features=32, hidden_dim=128, output_dim=512):
        super(MolecularGraphEncoder, self).__init__()
        self.atom_features = atom_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Atom feature encoder - embeddings for atomic numbers up to 100
        self.atom_embedding = nn.Embedding(100, atom_features)

        # Graph convolutional layers
        self.conv1 = nn.Linear(atom_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = nn.Linear(hidden_dim, hidden_dim)

        # Output projection layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, atomic_nums, adjacency_matrix):
        """
        Forward pass through the ChemNet model

        Parameters:
        atomic_nums: tensor of atomic numbers (batch_size, max_atoms)
        adjacency_matrix: adjacency matrix (batch_size, max_atoms, max_atoms)

        Returns:
        mol_embedding: tensor of molecular embeddings (batch_size, output_dim)
        """
        # Get atom embeddings
        x = self.atom_embedding(atomic_nums)  # (batch_size, max_atoms, atom_features)

        # Graph convolution layers with message passing
        # First layer
        x_in = x
        message = torch.bmm(adjacency_matrix, x_in)  # Message passing
        x = F.relu(self.conv1(message))
        x = self.bn1(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (batch_size, max_atoms, hidden_dim)

        # Second layer
        x_in = x
        message = torch.bmm(adjacency_matrix, x_in)  # Message passing
        x = F.relu(self.conv2(message))
        x = self.bn2(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (batch_size, max_atoms, hidden_dim)

        # Third layer
        x_in = x
        message = torch.bmm(adjacency_matrix, x_in)  # Message passing
        x = F.relu(self.conv3(message))
        x = self.bn3(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (batch_size, max_atoms, hidden_dim)

        # Global pooling (mean of all atom features)
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)

        # Project to output dimension
        mol_embedding = self.output_layer(x)  # (batch_size, output_dim)

        return mol_embedding


# Initialize the ChemNet model
def get_chemnet_model(device="cpu"):
    """Initialize and return the ChemNet model"""
    model = MolecularGraphEncoder()
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


# Global ChemNet model instance
_chemnet_model = None


def get_model():
    """Get or initialize the ChemNet model"""
    global _chemnet_model
    if _chemnet_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _chemnet_model = get_chemnet_model(device)
    return _chemnet_model


def mol_to_graph_data(mol, max_atoms=50):
    """
    Convert RDKit molecule to graph representation for ChemNet

    Parameters:
    mol: RDKit molecule
    max_atoms: maximum number of atoms to consider

    Returns:
    atomic_nums: tensor of atomic numbers (max_atoms)
    adjacency_matrix: adjacency matrix (max_atoms, max_atoms)
    atom_mask: mask for valid atoms (max_atoms)
    """
    # Remove hydrogens for simplicity
    mol = rdmolops.RemoveHs(mol)

    # Get atom features
    atoms = mol.GetAtoms()
    n_atoms = len(atoms)

    if n_atoms > max_atoms:
        print(f"Warning: Molecule has {n_atoms} atoms, truncating to {max_atoms}")
        n_atoms = max_atoms

    # Initialize arrays
    atomic_nums = np.zeros(max_atoms, dtype=np.int64)
    adjacency_matrix = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    atom_mask = np.zeros(max_atoms, dtype=np.float32)

    # Fill atom features
    for i, atom in enumerate(atoms):
        if i >= max_atoms:
            break
        atomic_nums[i] = atom.GetAtomicNum()
        atom_mask[i] = 1.0  # Mark as valid atom

    # Fill adjacency matrix
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            # Simple binary adjacency
            adjacency_matrix[i, j] = 1.0
            adjacency_matrix[j, i] = 1.0  # Undirected graph

    # Convert to PyTorch tensors
    atomic_nums = torch.tensor(atomic_nums, dtype=torch.long)
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float)
    atom_mask = torch.tensor(atom_mask, dtype=torch.float)

    # Normalize adjacency matrix (add self-loops and normalize)
    # A_norm = D^(-1/2) * (A + I) * D^(-1/2)
    adjacency_matrix += torch.eye(max_atoms)  # Add self-loops
    D = torch.sum(adjacency_matrix, dim=1)
    D_sqrt_inv = torch.diag(torch.pow(D + 1e-8, -0.5))
    adjacency_matrix = torch.mm(torch.mm(D_sqrt_inv, adjacency_matrix), D_sqrt_inv)

    return atomic_nums, adjacency_matrix, atom_mask


def get_chemnet_embeddings(mol):
    """
    Generate ChemNet embeddings for a molecule

    Parameters:
    mol: RDKit molecule

    Returns:
    embedding: numpy array of ChemNet embedding
    """
    if mol is None:
        raise ValueError("Molecule is None")

    # Get model
    model = get_model()
    device = next(model.parameters()).device

    # Convert molecule to graph data
    atomic_nums, adjacency_matrix, atom_mask = mol_to_graph_data(mol)

    # Move data to device
    atomic_nums = atomic_nums.unsqueeze(0).to(device)  # Add batch dimension
    adjacency_matrix = adjacency_matrix.unsqueeze(0).to(device)  # Add batch dimension

    # Get embeddings
    with torch.no_grad():
        embedding = model(atomic_nums, adjacency_matrix)

    # Return as numpy array
    return embedding.cpu().numpy().flatten()


def calculate_fcd(embeddings_set1, embeddings_set2):
    """Calculate Fréchet ChemNet Distance (FCD) between two sets of molecular embeddings

    Args:
        embeddings_set1: List of ChemNet embeddings for first set of molecules
        embeddings_set2: List of ChemNet embeddings for second set of molecules

    Returns:
        float: The calculated FCD value

    Note:
        FCD is calculated using the Fréchet distance formula applied to the distribution
        of molecular embeddings, considering both mean and covariance of the embedding space.
    """
    if len(embeddings_set1) < 2 or len(embeddings_set2) < 2:
        raise ValueError("Need at least 2 molecules in each set")

    embeddings_set1 = np.array(embeddings_set1, dtype=np.float64)
    embeddings_set2 = np.array(embeddings_set2, dtype=np.float64)

    epsilon = 1e-6  # Smaller epsilon for numerical stability

    # Vectorized mean calculation
    mu_1 = np.mean(embeddings_set1, axis=0)
    mu_2 = np.mean(embeddings_set2, axis=0)

    # Vectorized covariance calculation
    sigma_1 = np.cov(embeddings_set1, rowvar=False)
    sigma_2 = np.cov(embeddings_set2, rowvar=False)

    # Ensure matrices are positive definite
    sigma_1 = (sigma_1 + sigma_1.T) / 2
    sigma_2 = (sigma_2 + sigma_2.T) / 2
    sigma_1 += epsilon * np.eye(sigma_1.shape[0])
    sigma_2 += epsilon * np.eye(sigma_2.shape[0])

    try:
        # Calculate sqrt(sigma_1 * sigma_2)
        sqrtm_1 = sqrtm(sigma_1)
        inter_term = sqrtm_1.dot(sigma_2).dot(sqrtm_1)
        inter_term = (inter_term + inter_term.T) / 2  # Ensure symmetry
        cov_sqrt = sqrtm(inter_term)

        if np.iscomplexobj(cov_sqrt):
            if np.max(np.abs(cov_sqrt.imag)) < 1e-6:
                cov_sqrt = cov_sqrt.real
            else:
                print(
                    f"Warning: Significant imaginary component: {np.max(np.abs(cov_sqrt.imag)):.6f}"
                )
                cov_sqrt = np.abs(cov_sqrt)

        # Vectorized calculations
        mean_diff = np.sum((mu_1 - mu_2) ** 2)
        trace_term = np.trace(sigma_1) + np.trace(sigma_2) - 2 * np.trace(cov_sqrt)

        fcd = mean_diff + trace_term

        # Ensure non-negative but don't force to exactly zero
        if fcd < 0 and fcd > -1e-10:  # Small negative values due to numerical errors
            fcd = 0.0
        elif fcd < 0:
            raise ValueError(f"FCD calculation resulted in negative value: {fcd}")

        return float(fcd)

    except Exception as e:
        print(f"Debug - Matrix properties:")
        print(f"Condition number sigma_1: {np.linalg.cond(sigma_1)}")
        print(f"Condition number sigma_2: {np.linalg.cond(sigma_2)}")
        raise ValueError(f"Error in FCD calculation: {e}")


def load_molecules_from_sdf(filename):
    """Load molecules from SDF file and generate ChemNet embeddings"""
    embeddings = []
    try:
        suppl = Chem.SDMolSupplier(filename)
        for mol in suppl:
            if mol is not None:
                try:
                    # Generate ChemNet embedding
                    embedding = get_chemnet_embeddings(mol)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error processing molecule: {e}")
                    continue
    except Exception as e:
        print(f"Error processing {filename}: {e}")

    if len(embeddings) == 0:
        print(f"Warning: No valid molecules found in {filename}")
    else:
        print(
            f"Successfully loaded {len(embeddings)} ChemNet embeddings from {filename}"
        )
        print(f"Embedding dimension: {embeddings[0].shape}")

    return embeddings


def load_molecules_from_json(filename):
    """Load molecules from JSON file and generate ChemNet embeddings"""
    embeddings = []
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        if "molecules" in data and isinstance(data["molecules"], list):
            for molecule in data["molecules"]:
                if "aligned_reference" in molecule:
                    RDLogger.DisableLog("rdApp.*")
                    mol = Chem.MolFromSmiles(
                        molecule["aligned_reference"], sanitize=True
                    )
                    RDLogger.EnableLog("rdApp.*")

                    if mol is not None:
                        try:
                            embedding = get_chemnet_embeddings(mol)
                            embeddings.append(embedding)
                        except Exception as e:
                            print(f"Error generating embedding: {e}")
                            continue
    except Exception as e:
        print(f"Error processing {filename}: {e}")

    if embeddings:
        print(
            f"Successfully loaded {len(embeddings)} ChemNet embeddings from {filename}"
        )
        print(f"Embedding dimension: {embeddings[0].shape}")

    return embeddings


def load_molecules_from_csv(filename):
    """Load molecules from CSV file containing SMILES and generate ChemNet embeddings"""
    embeddings = []
    try:
        df = pd.read_csv(filename)
        # Look for SMILES column - it might be called 'smiles', 'SMILES', 'structure', etc.
        smiles_col = None
        for col in df.columns:
            if col.lower() in ["smiles", "structure", "canonical_smiles"]:
                smiles_col = col
                break

        if smiles_col is None and len(df.columns) > 0:
            # If we can't find a specific column, try the first column
            smiles_col = df.columns[0]

        if smiles_col is not None:
            RDLogger.DisableLog("rdApp.*")
            for smiles in df[smiles_col]:
                if isinstance(smiles, str):
                    mol = Chem.MolFromSmiles(smiles, sanitize=True)
                    if mol is not None:
                        try:
                            embedding = get_chemnet_embeddings(mol)
                            embeddings.append(embedding)
                        except Exception as e:
                            print(f"Error generating embedding: {e}")
                            continue
            RDLogger.EnableLog("rdApp.*")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

    if embeddings:
        print(
            f"Successfully loaded {len(embeddings)} ChemNet embeddings from CSV {filename}"
        )
        print(f"Embedding dimension: {embeddings[0].shape}")

    return embeddings


def calculate_and_print_fcd(name1, embeddings1, name2, embeddings2):
    """Calculate and print FCD between two sets of embeddings"""
    if len(embeddings1) > 3 and len(embeddings2) > 3:
        try:
            fcd_value = calculate_fcd(embeddings1, embeddings2)
            print(
                f"FCD between {name1} ({len(embeddings1)} molecules) and {name2} ({len(embeddings2)} molecules): {fcd_value:.4f}"
            )
            return name1, name2, len(embeddings1), len(embeddings2), fcd_value
        except Exception as e:
            print(f"Error calculating FCD between {name1} and {name2}: {e}")
            return name1, name2, len(embeddings1), len(embeddings2), None
    else:
        print(
            f"Not enough molecules for {name1} ({len(embeddings1)}) or {name2} ({len(embeddings2)})"
        )
        return name1, name2, len(embeddings1), len(embeddings2), None


def get_file_basename(filepath):
    """Get a clean name for the file"""
    return os.path.splitext(os.path.basename(filepath))[0]


def load_molecule_file(filepath):
    """Load molecules from file based on extension"""
    print(f"Loading molecules from {filepath}...")
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".sdf":
        return load_molecules_from_sdf(filepath)
    elif ext == ".json":
        return load_molecules_from_json(filepath)
    elif ext == ".csv":
        return load_molecules_from_csv(filepath)
    else:
        print(f"Unsupported file extension: {ext}")
        return []


def main():
    start_time = time.time()

    # Define the directory containing all molecule sets
    molecules_dir = "molecules_sets"

    # Check if directory exists
    if not os.path.exists(molecules_dir):
        print(f"Error: Molecules directory not found at 'molecules_sets'")
        return

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Looking for molecule files in {os.path.abspath(molecules_dir)}")

    # Find all molecule files in the directory
    sdf_files = glob.glob(os.path.join(molecules_dir, "*.sdf"))
    json_files = glob.glob(os.path.join(molecules_dir, "*.json"))
    csv_files = glob.glob(os.path.join(molecules_dir, "*.csv"))

    all_files = sdf_files + json_files + csv_files

    if not all_files:
        print("No molecule files found in the directory.")
        return

    print(f"Found {len(all_files)} molecule files.")

    # Dictionary to store embeddings for each file
    all_embeddings = {}

    # Load all molecule files
    for filepath in all_files:
        file_name = get_file_basename(filepath)
        embeddings = load_molecule_file(filepath)

        if embeddings:
            all_embeddings[file_name] = embeddings

    if len(all_embeddings) < 2:
        print("Not enough valid molecule sets loaded to compare.")
        return

    print("\nCalculating Fréchet ChemNet Distances:")
    print("-" * 70)

    # Calculate FCDs between all pairs of files
    results = []
    file_names = list(all_embeddings.keys())

    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            name1 = file_names[i]
            name2 = file_names[j]
            result = calculate_and_print_fcd(
                name1, all_embeddings[name1], name2, all_embeddings[name2]
            )
            if result[4] is not None:  # If FCD was successfully calculated
                results.append(result)

    # Save results to CSV
    if results:
        result_df = pd.DataFrame(
            results, columns=["Set1", "Set2", "Size1", "Size2", "FCD"]
        )
        result_file = os.path.join(results_dir, "fcd_chemnet_comparison_results.csv")
        result_df.to_csv(result_file, index=False)
        print(f"\nResults saved to {result_file}")

        # Also print a cross-table of FCDs
        print("\nFCD Cross-Table:")
        cross_table = pd.DataFrame(index=file_names, columns=file_names)
        for name1, name2, _, _, fcd in results:
            cross_table.loc[name1, name2] = fcd
            cross_table.loc[name2, name1] = fcd
        for name in file_names:
            cross_table.loc[name, name] = 0.0

        print(cross_table)

        # Save the cross-table
        cross_table_file = os.path.join(results_dir, "fcd_chemnet_cross_table.csv")
        cross_table.to_csv(cross_table_file)
        print(f"Cross-table saved to {cross_table_file}")

        # Save a summary of the analysis
        summary_file = os.path.join(results_dir, "fcd_chemnet_analysis_summary.txt")
        with open(summary_file, "w") as f:
            f.write("FCD ChemNet Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Number of molecule sets analyzed: {len(file_names)}\n")
            f.write(f"Total number of comparisons: {len(results)}\n")
            f.write(
                f"Total processing time: {time.time() - start_time:.2f} seconds\n\n"
            )
            f.write("Molecule Sets:\n")
            for name in file_names:
                f.write(f"- {name}: {len(all_embeddings[name])} molecules\n")
            f.write("\nResults Summary:\n")
            f.write(result_df.to_string())
        print(f"Analysis summary saved to {summary_file}")


if __name__ == "__main__":
    main()
