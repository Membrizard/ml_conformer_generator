import onnxruntime
import torch
import random

session = onnxruntime.InferenceSession('./100_steps_edm_moi_chembl_15_39.onnx')

input_shapes = [x.shape for x in session.get_inputs()]
input_names = [x.name for x in session.get_inputs()]

output_names = [x.name for x in session.get_outputs()]


def prepare_dummy_input(device):
    reference_context = torch.tensor(
        [53.6424, 108.3042, 151.4399], dtype=torch.float32, device=device
    )
    context_norms = {
        "mean": torch.tensor(
            [105.0766, 473.1938, 537.4675], dtype=torch.float32, device=device
        ),
        "mad": torch.tensor(
            [52.0409, 219.7475, 232.9718], dtype=torch.float32, device=device
        ),
    }

    n_samples = 2
    min_n_nodes = 15
    max_n_nodes = 20
    f_max_n_nodes = 39

    # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples
    nodesxsample = []

    for n in range(n_samples):
        nodesxsample.append(random.randint(min_n_nodes, max_n_nodes))

    nodesxsample = torch.tensor(nodesxsample)

    batch_size = nodesxsample.size(0)

    node_mask = torch.zeros(batch_size, f_max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0: nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(
        batch_size * f_max_n_nodes * f_max_n_nodes, 1
    ).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    normed_context = (
        (reference_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    batch_context = normed_context.unsqueeze(0).repeat(batch_size, 1)

    batch_context = (
        batch_context.unsqueeze(1).repeat(1, f_max_n_nodes, 1) * node_mask
    )
    return node_mask.numpy(), edge_mask.numpy(), batch_context.numpy()


inputs = prepare_dummy_input("cpu")

print(input_names)
print(input_shapes)
print(output_names)

out = session.run(["x", "h"], {"node_mask": inputs[0], "edge_mask": inputs[1], "context": inputs[2]})

print(out)

