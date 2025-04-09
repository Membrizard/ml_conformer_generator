import numpy as np
import pyvista as pv
from rdkit import Chem
import trimesh


def align_to_principal_axes(points):
    """Align points to their principal axes using PCA"""
    # Center the points
    center = np.mean(points, axis=0)
    centered_points = points - center

    # Calculate covariance matrix
    covariance_matrix = np.cov(centered_points.T)

    # Get eigenvectors (principal axes)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Rotate points to align with principal axes
    aligned_points = np.dot(centered_points, eigenvectors)

    return aligned_points, eigenvectors, center


def visualize_molecule_and_stl(stl_file, sdf_file, mol_idx=0):
    # Create plotter
    plotter = pv.Plotter()

    # Load molecule
    suppl = Chem.SDMolSupplier(sdf_file)
    mol = suppl[mol_idx]

    # Get atom coordinates
    conf = mol.GetConformer()
    atom_coords = []
    atom_types = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_coords.append([pos.x, pos.y, pos.z])
        atom_types.append(atom.GetSymbol())
    atom_coords = np.array(atom_coords)

    # Align molecule to its principal axes
    mol_aligned, mol_rotation, mol_center = align_to_principal_axes(atom_coords)

    # Load STL
    stl_mesh = trimesh.load_mesh(stl_file)
    stl_vertices = stl_mesh.vertices

    # Align STL to its principal axes
    stl_aligned, stl_rotation, stl_center = align_to_principal_axes(stl_vertices)

    # Create aligned STL mesh
    aligned_stl = trimesh.Trimesh(vertices=stl_aligned, faces=stl_mesh.faces)

    # Convert to PyVista mesh
    vertices = aligned_stl.vertices
    faces = aligned_stl.faces
    pv_mesh = pv.PolyData(vertices, np.insert(faces, 0, 3, axis=1))
    plotter.add_mesh(pv_mesh, color="lightblue", opacity=0.5)

    # Print alignment information
    print("\nAlignment Information:")
    print(f"Molecule rotation matrix:\n{mol_rotation}")
    print(f"STL rotation matrix:\n{stl_rotation}")
    print(f"Molecule center: {mol_center}")
    print(f"STL center: {stl_center}")

    # Function to add molecule
    def add_molecule(coords):
        # Add atoms as spheres
        colors = {
            "C": "gray",
            "N": "blue",
            "O": "red",
            "S": "yellow",
            "P": "orange",
            "F": "green",
            "Cl": "green",
            "Br": "brown",
            "I": "purple",
            "H": "white",
        }

        # Add atoms
        for coord, atom_type in zip(coords, atom_types):
            sphere = pv.Sphere(radius=0.3, center=coord)
            color = colors.get(atom_type, "gray")
            plotter.add_mesh(sphere, color=color)

        # Add bonds
        for bond in mol.GetBonds():
            id1 = bond.GetBeginAtomIdx()
            id2 = bond.GetEndAtomIdx()
            pt1 = coords[id1]
            pt2 = coords[id2]

            # Create cylinder for bond
            direction = pt2 - pt1
            length = np.linalg.norm(direction)
            cylinder = pv.Cylinder(
                center=(pt1 + pt2) / 2, direction=direction, radius=0.1, height=length
            )
            plotter.add_mesh(cylinder, color="gray")

    # Add aligned molecule
    add_molecule(mol_aligned)

    # Add coordinate axes
    plotter.add_axes()

    # Show plot
    plotter.show()


if __name__ == "__main__":
    visualize_molecule_and_stl(
        "6q8k_pocket_a 2.stl", "37_39_6q8k_generated_molecules_2 1.sdf"
    )
