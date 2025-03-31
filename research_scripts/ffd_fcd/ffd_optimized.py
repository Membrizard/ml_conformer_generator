"""
This module implements an optimized version of the Fréchet Fingerprint Distance (FFD) calculation.
It includes vectorized operations and parallel processing for improved performance.
"""

import gc
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from scipy.linalg import sqrtm


def get_morgan_fingerprints(mol, radius=2, nBits=2048):
    """Generate Morgan fingerprints for a molecule using RDKit's Morgan fingerprint generator

    Args:
        mol: RDKit molecule object
        radius: The radius of the Morgan fingerprint (default: 2)
        nBits: The number of bits in the fingerprint (default: 2048)

    Returns:
        numpy.ndarray: The Morgan fingerprint as a dense binary array
    """
    try:
        # Generate Morgan fingerprint using MorganGenerator
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nBits
        )
        fp = morgan_gen.GetFingerprintAsNumPy(mol)
        # Convert to float64 but skip normalization to preserve differences
        return fp.astype(np.float64)
    except Exception as e:
        raise ValueError(f"Error generating fingerprint: {e}")


def calculate_ffd(embeddings_set1, embeddings_set2):
    """Calculate Fréchet Fingerprint Distance (FFD) between two sets of molecular fingerprints

    Args:
        embeddings_set1: List of Morgan fingerprints for first set of molecules
        embeddings_set2: List of Morgan fingerprints for second set of molecules

    Returns:
        float: The calculated FFD value
    """
    if len(embeddings_set1) < 2 or len(embeddings_set2) < 2:
        raise ValueError("Need at least 2 molecules in each set")

    embeddings_set1 = np.array(embeddings_set1, dtype=np.float64)
    embeddings_set2 = np.array(embeddings_set2, dtype=np.float64)

    epsilon = 1e-6

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
        inter_term = (inter_term + inter_term.T) / 2
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

        ffd = mean_diff + trace_term

        if ffd < 0 and ffd > -1e-10:
            ffd = 0.0
        elif ffd < 0:
            raise ValueError(f"FFD calculation resulted in negative value: {ffd}")

        return float(ffd)

    except Exception as e:
        print(f"Debug - Matrix properties:")
        print(f"Condition number sigma_1: {np.linalg.cond(sigma_1)}")
        print(f"Condition number sigma_2: {np.linalg.cond(sigma_2)}")
        raise ValueError(f"Error in FFD calculation: {e}")


def process_molecule_batch(molecules, radius=2, nBits=2048):
    """Process a batch of molecules to generate fingerprints in parallel"""
    fingerprints = []
    for mol in molecules:
        if mol is not None:
            try:
                fp = get_morgan_fingerprints(mol, radius, nBits)
                fingerprints.append(fp)
            except Exception as e:
                print(f"Error processing molecule: {e}")
                continue
    return fingerprints


def load_molecules_from_sdf(filename, batch_size=100):
    """Load molecules from SDF file with parallel processing"""
    embeddings = []
    try:
        suppl = Chem.SDMolSupplier(filename)
        molecules = []

        # First, collect all valid molecules
        for mol in suppl:
            if mol is not None:
                molecules.append(mol)

        # Process molecules in batches using parallel processing
        with ProcessPoolExecutor() as executor:
            process_func = partial(process_molecule_batch)
            for i in range(0, len(molecules), batch_size):
                batch = molecules[i : i + batch_size]
                batch_fps = executor.submit(process_func, batch).result()
                embeddings.extend(batch_fps)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

    if len(embeddings) == 0:
        print(f"Warning: No valid molecules/fingerprints found in {filename}")
    else:
        print(f"Successfully loaded {len(embeddings)} fingerprints from {filename}")
        print(f"Fingerprint shape: {embeddings[0].shape}")
        print(f"Sample fingerprint non-zero bits: {np.sum(embeddings[0])}")

    return embeddings


def load_molecules_from_json(filename, batch_size=100):
    """Load molecules from JSON file with parallel processing"""
    embeddings = []
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        if "molecules" in data and isinstance(data["molecules"], list):
            molecules = []
            RDLogger.DisableLog("rdApp.*")

            # First, collect all valid molecules
            for molecule in data["molecules"]:
                if "aligned_reference" in molecule:
                    mol = Chem.MolFromSmiles(
                        molecule["aligned_reference"], sanitize=True
                    )
                    if mol is not None:
                        molecules.append(mol)

            RDLogger.EnableLog("rdApp.*")

            # Process molecules in batches using parallel processing
            with ProcessPoolExecutor() as executor:
                process_func = partial(process_molecule_batch)
                for i in range(0, len(molecules), batch_size):
                    batch = molecules[i : i + batch_size]
                    batch_fps = executor.submit(process_func, batch).result()
                    embeddings.extend(batch_fps)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

    if embeddings:
        print(f"Successfully loaded {len(embeddings)} fingerprints from {filename}")
        print(f"Fingerprint shape: {embeddings[0].shape}")

    return embeddings


def load_molecules_from_csv(filename, batch_size=100):
    """Load molecules from CSV file with parallel processing"""
    embeddings = []
    try:
        df = pd.read_csv(filename)
        smiles_col = None
        for col in df.columns:
            if col.lower() in ["smiles", "structure", "canonical_smiles"]:
                smiles_col = col
                break

        if smiles_col is None and len(df.columns) > 0:
            smiles_col = df.columns[0]

        if smiles_col is not None:
            molecules = []
            RDLogger.DisableLog("rdApp.*")

            # First, collect all valid molecules
            for smiles in df[smiles_col]:
                if isinstance(smiles, str):
                    mol = Chem.MolFromSmiles(smiles, sanitize=True)
                    if mol is not None:
                        molecules.append(mol)

            RDLogger.EnableLog("rdApp.*")

            # Process molecules in batches using parallel processing
            with ProcessPoolExecutor() as executor:
                process_func = partial(process_molecule_batch)
                for i in range(0, len(molecules), batch_size):
                    batch = molecules[i : i + batch_size]
                    batch_fps = executor.submit(process_func, batch).result()
                    embeddings.extend(batch_fps)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

    if embeddings:
        print(f"Successfully loaded {len(embeddings)} fingerprints from CSV {filename}")
        print(f"Fingerprint shape: {embeddings[0].shape}")

    return embeddings


def calculate_and_print_ffd(name1, embeddings1, name2, embeddings2):
    """Calculate and print FFD between two sets of embeddings"""
    if len(embeddings1) > 3 and len(embeddings2) > 3:
        try:
            ffd_value = calculate_ffd(embeddings1, embeddings2)
            print(
                f"FFD between {name1} ({len(embeddings1)} molecules) and {name2} ({len(embeddings2)} molecules): {ffd_value:.4f}"
            )
            return name1, name2, len(embeddings1), len(embeddings2), ffd_value
        except Exception as e:
            print(f"Error calculating FFD between {name1} and {name2}: {e}")
            return name1, name2, len(embeddings1), len(embeddings2), None
    else:
        print(
            f"Not enough molecules for {name1} ({len(embeddings1)}) or {name2} ({len(embeddings2)})"
        )
        return name1, name2, len(embeddings1), len(embeddings2), None


def get_file_basename(filepath):
    """Get a clean name for the file"""
    return os.path.splitext(os.path.basename(filepath))[0]


def load_molecule_file(filepath, batch_size=100):
    """Load molecules from file based on extension"""
    print(f"Loading molecules from {filepath}...")
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".sdf":
        return load_molecules_from_sdf(filepath, batch_size)
    elif ext == ".json":
        return load_molecules_from_json(filepath, batch_size)
    elif ext == ".csv":
        return load_molecules_from_csv(filepath, batch_size)
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

    print("\nCalculating Fréchet Fingerprint Distances:")
    print("-" * 70)

    # Calculate FFDs between all pairs of files
    results = []
    file_names = list(all_embeddings.keys())

    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            name1 = file_names[i]
            name2 = file_names[j]
            result = calculate_and_print_ffd(
                name1, all_embeddings[name1], name2, all_embeddings[name2]
            )
            if result[4] is not None:  # If FFD was successfully calculated
                results.append(result)

    # Save results to CSV
    if results:
        result_df = pd.DataFrame(
            results, columns=["Set1", "Set2", "Size1", "Size2", "FFD"]
        )
        result_file = os.path.join(results_dir, "ffd_comparison_results.csv")
        result_df.to_csv(result_file, index=False)
        print(f"\nResults saved to {result_file}")

        # Also print a cross-table of FFDs
        print("\nFFD Cross-Table:")
        cross_table = pd.DataFrame(index=file_names, columns=file_names)
        for name1, name2, _, _, ffd in results:
            cross_table.loc[name1, name2] = ffd
            cross_table.loc[name2, name1] = ffd
        for name in file_names:
            cross_table.loc[name, name] = 0.0

        print(cross_table)

        # Save the cross-table
        cross_table_file = os.path.join(results_dir, "ffd_cross_table.csv")
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
