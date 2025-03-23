import torch.jit
import random
from ml_conformer_generator.ml_conformer_generator.compilable_egnn import EGNNDynamics
from ml_conformer_generator.ml_conformer_generator.compilable_equivariant_diffusion import EquivariantDiffusion
from ml_conformer_generator.ml_conformer_generator.utils import get_context_shape
from rdkit import Chem
from rdkit.Chem import rdDistGeom

device = "cuda"
net_dynamics = EGNNDynamics(
            in_node_nf=9,
            context_node_nf=3,
            hidden_nf=420,
            device=device,
        )

generative_model = EquivariantDiffusion(
            dynamics=net_dynamics,
            in_node_nf=8,
            timesteps=1000,
            noise_precision=1e-5,
        )

generative_model.to(device)

edm_weights = "./ml_conformer_generator/ml_conformer_generator/weights/compilable_weights/compilable_edm_moi_chembl_15_39.weights"
generative_model.load_state_dict(
            torch.load(
                edm_weights,
                map_location=device,
            )
        )


generative_model.eval()

compiled_model = torch.jit.script(generative_model)


def prepare_dummy_input(
        smiles: str = 'Cc1cccc(N2CCC[C@H]2c2ccncn2)n1',
        n_samples=2,
        variance=2,
        device="cpu",
):
    """
    """
    ref_mol = Chem.MolFromSmiles(smiles)
    rdDistGeom.EmbedMolecule(ref_mol, forceTol=0.001, randomSeed=12)

    context_norms = {
        "mean": torch.tensor([105.0766, 473.1938, 537.4675]),
        "mad": torch.tensor([52.0409, 219.7475, 232.9718]),
    }

    reference_conformer = Chem.RemoveHs(ref_mol)
    ref_n_atoms = reference_conformer.GetNumAtoms()
    conf = reference_conformer.GetConformer()
    ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    # move coord to center
    virtual_com = torch.mean(ref_coord, dim=0)
    ref_coord = ref_coord - virtual_com

    ref_context, aligned_coord = get_context_shape(ref_coord)

    # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples
    nodesxsample = []

    min_n_nodes = ref_n_atoms - variance
    max_n_nodes = ref_n_atoms + variance

    # Make sure that number of atoms of generated samples is within requested range
    if min_n_nodes < 15:
        min_n_nodes = 15

    if max_n_nodes > 39:
        max_n_nodes = 39

    for n in range(n_samples):
        nodesxsample.append(random.randint(min_n_nodes, max_n_nodes))

    nodesxsample = torch.tensor(nodesxsample)

    batch_size = nodesxsample.size(0)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0: nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(
        device
    )
    node_mask = node_mask.unsqueeze(2).to(device)

    normed_context = (
            (ref_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    batch_context = normed_context.unsqueeze(0).repeat(batch_size, 1)

    batch_context = batch_context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask

    return n_samples, max_n_nodes, node_mask.to(device), edge_mask.to(device), batch_context.to(device)


dummy_input = prepare_dummy_input(device=device)

# Dummy input data for all arguments - Equivariant Diffusion
# n_samples = 1
# n_nodes = 2
# node_mask = torch.ones((1, 2, 1), dtype=torch.float32, device=device)
# edge_mask = torch.zeros((4, 1), dtype=torch.float32, device=device)
# context = torch.zeros((1, 2, 3), dtype=torch.float32, device=device)
#
# # dummy_input = (elements, dist_mat, adj_mat)
# dummy_input = (n_samples, n_nodes, node_mask, edge_mask, context)

# Exporting to ONNX
torch.onnx.export(
        compiled_model,
        dummy_input,  # Tuple of inputs
        "moi_edm_chembl_15_39.onnx",
        do_constant_folding=True,
        opset_version=18,
        export_params=True,
        input_names=["n_samples", "n_nodes", "node_mask", "edge_mask", "context"],
        output_names=["x", "h"],
        dynamic_axes={
                      "node_mask": {0: "batch_size", 1: "num_nodes"},
                      "edge_mask": {0: "num_edges"},
                      "context": {0: "batch_size", 1: "num_nodes"},
                      "x": {0: "batch_size", 1: "num_nodes"},
                      "h": {0: "batch_size", 1: "num_nodes"},
        },
        verbose=True,
    )

