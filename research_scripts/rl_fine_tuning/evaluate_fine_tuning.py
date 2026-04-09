import torch
from rdkit import Chem
from src.mlconfgen import MLConformerGenerator, evaluate_samples
from src.mlconfgen.utils import samples_to_rdkit_mol, ATOM_DECODER, prepare_adj_mat_seer_input, redefine_bonds, standardize_mol


def edm_sampler_fn(generator, task: dict,) -> list[Chem.Mol]:
    ref_context, ref_n_atoms, fixed_fragment = generator.prepare_inputs(
        reference_conformer=task["reference_conformer"],
        fixed_fragment=None,
        reference_context=None,
        n_atoms=None,
    )

    min_n_nodes = ref_n_atoms - task["variance"]
    max_n_nodes = ref_n_atoms + task["variance"]
    return generator.edm_samples(
        reference_context=ref_context,
        n_samples=task["n_samples"],
        min_n_nodes=min_n_nodes,
        max_n_nodes=max_n_nodes,
        raw_output=True,
    )


def set_mol_props(mol, ref_name, score_value, score_name, valid_flag):
    mol.SetProp("reference_mol", ref_name)
    mol.SetProp("score_value", str(score_value))
    mol.SetProp("score_name", score_name)
    mol.SetProp("validity", str(valid_flag))
    return None


@torch.no_grad()
def evaluate(
             bl_generator,
             ft_generator,
             score_fn: dict,
             task: dict,
             agent_writer,
             baseline_writer,




             ) -> dict[str, float]:

    ref_name = task["reference_conformer"].GetProp("_Name")

    x, h, node_mask, edge_mask = edm_sampler_fn(bl_generator, task)

    _x, _h, edm_adapter_log_probs, edm_aux = ft_generator.edm_adapter(
        x=x,
        h=h,
        node_mask=node_mask,
        edge_mask=edge_mask,
        sample=False,
    )

    ft_samples = samples_to_rdkit_mol(
        positions=_x, one_hot=_h, node_mask=node_mask, atom_decoder=ATOM_DECODER
    )

    baseline_samples = samples_to_rdkit_mol(
        positions=x, one_hot=h, node_mask=node_mask, atom_decoder=ATOM_DECODER
    )

    (
        el_batch,
        dm_batch,
        b_adj_mat_batch,
        canonicalised_samples,
    ) = prepare_adj_mat_seer_input(
        mols=ft_samples,
        dimension=ft_generator.model.dimension,
        device=ft_generator.device,
    )

    (
        bl_el_batch,
        bl_dm_batch,
        bl_b_adj_mat_batch,
        bl_canonicalised_samples,
    ) = prepare_adj_mat_seer_input(
        mols=baseline_samples,
        dimension=bl_generator.model.dimension,
        device=bl_generator.device,
    )

    agent_adj_mat_batch = ft_generator.adj_mat_seer(
        elements=el_batch,
        dist_mat=dm_batch,
        adj_mat=b_adj_mat_batch,
    )

    baseline_adj_mat_batch = bl_generator.adj_mat_seer(
        elements=bl_el_batch,
        dist_mat=bl_dm_batch,
        adj_mat=bl_b_adj_mat_batch,
    )

    agent_scores = []
    baseline_scores = []

    agent_valid_flags = []
    baseline_valid_flags = []

    agent_adj_mat_batch.to("cpu")
    baseline_adj_mat_batch.to("cpu")

    for i, base_mol in enumerate(canonicalised_samples):
        agent_adj_mat = agent_adj_mat_batch[i]
        baseline_adj_mat = baseline_adj_mat_batch[i]
        _baseline_mol = bl_canonicalised_samples[i]

        agent_mol = redefine_bonds(mol=base_mol, adj_mat=agent_adj_mat)
        baseline_mol = redefine_bonds(mol=_baseline_mol, adj_mat=baseline_adj_mat)

        agent_score, agent_valid_value = _eval_op(
            agent_mol, agent_adj_mat, score_fn["function"]
        )
        baseline_score, baseline_valid_value = _eval_op(
            baseline_mol, baseline_adj_mat, score_fn["function"]
        )

        # Log everything to sdf
        set_mol_props(agent_mol, ref_name, agent_score, score_fn["name"], agent_valid_value)
        set_mol_props(baseline_mol, ref_name, baseline_score, score_fn["name"], baseline_valid_value)

        agent_writer.write(agent_mol)
        baseline_writer.write(baseline_mol)

        agent_scores.append(agent_score)
        baseline_scores.append(baseline_score)

        agent_valid_flags.append(agent_valid_value)
        baseline_valid_flags.append(baseline_valid_value)

    agent_scores_t = torch.tensor(agent_scores, dtype=torch.float32)
    baseline_scores_t = torch.tensor(baseline_scores, dtype=torch.float32)

    agent_valid_t = torch.tensor(agent_valid_flags, dtype=torch.float32)
    baseline_valid_t = torch.tensor(baseline_valid_flags, dtype=torch.float32)

    f_agent_score = agent_scores_t.mean().item()
    f_baseline_score = baseline_scores_t.mean().item()

    baseline_imporv = f_agent_score - f_baseline_score

    return {
        "eval_agent_scores_mean": agent_scores_t.mean().item(),
        "eval_baseline_scores_mean": baseline_scores_t.mean().item(),
        "eval_agent_valid_rate": agent_valid_t.mean().item(),
        "eval_baseline_valid_rate": baseline_valid_t.mean().item(),
        "eval_improve_mean": baseline_imporv,
    }


def _eval_op(base_mol: Chem.Mol, adj_mat: torch.Tensor, score_fn) -> tuple:
    mol = redefine_bonds(mol=base_mol, adj_mat=adj_mat)
    valid_value = 1.0 if is_valid_mol(mol) else 0.0
    score = 0
    if valid_value == 1:
        score = score_fn(mol)

    return score, valid_value


def is_valid_mol(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    try:
        test_mol = Chem.Mol(mol)
        Chem.SanitizeMol(test_mol)
        return True
    except Exception:
        return False


def reinvent_score(mols: list[Chem.Mol | None]) -> list[float]:
    return None

# -------------------------------------------

# Load a Reference conformer

# Aim 20-50k samples per fine-tune

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
else:
    device = torch.device("cpu")

# 20, 25 and 30 heavy atom mol references from 1000_ccdc_validation_set
# t_names = ["CHEMBL63_P10000009", "CHEMBL3912858_P10000013", "CHEMBL2177159_P10000081"]
N_SAMPLES = 100
N_TASKS = 10
VARIANCE = 1

REF_MOLS = Chem.SDMolSupplier("")
DIFFUSION_STEPS = 100
# SCORES = [
#           {"name": "validity", "function": validity_score},
#           {"name": "shape", "function": shape_score},
#           {"name": "shape_and_color", "function": shape_and_color_score},
#           ]

SCORE_NAMES = ["validity",  "reinvent"]

FT_SAMPLES_WRITER = Chem.SDWriter("")
BL_SAMPLES_WRITER = Chem.SDWriter("")

TASK = {
    "reference_conformer": None,
    "variance": 1,
    "n_samples": 100,
}

for r_mol in REF_MOLS:
    print("Starting the Cycle for {Ref Name}")

    c_task = {
        "reference_conformer": r_mol,
        "variance": VARIANCE,
        "n_samples": N_SAMPLES,
    }
    ref_name = r_mol.GetProp("_Name")

    for score_name in SCORE_NAMES:
            print("Running Fine Tuning with {diffusion steps} {score fn name}")
            ft_checkpoints_dir = f"./{ref_name}_{DIFFUSION_STEPS}_{score_fn['name']}"
            ft_samples_writer = Chem.SDWriter(f"{ft_checkpoints_dir}/fine_tuned_samples.sdf")
            bl_samples_writer = Chem.SDWriter(f"{ft_checkpoints_dir}/baseline_samples.sdf")
            logfile = f"{ft_checkpoints_dir}/evaluation_report.log"

            match score_name:
                case "validity":
                    score_fn = validity_score
                case "shape":
                    def shape_score(mol: Chem.Mol | None):
                        try:
                            mol = standardize_mol(mol, optimize_geometry=True, ifm_mode=True)
                            ref_mb, scores = evaluate_samples(r_mol, [mol])
                            return scores[0]['shape_tanimoto']
                        except:
                            return 0

                    score_fn = shape_score
                case "shape_and_color":
                    def shape_and_color_score(mol: Chem.Mol | None):
                        try:
                            mol = standardize_mol(mol, optimize_geometry=True, ifm_mode=True)
                            ref_mb, scores = evaluate_samples(r_mol, [mol])
                            return scores[0]['shape_tanimoto']
                        except:
                            return 0
                    score_fn = shape_and_color_score

            ft_generator = MLConformerGenerator(
                edm_weights="./edm_moi_chembl_15_39.pt",
                adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
                device=device,
                diffusion_steps=DIFFUSION_STEPS,
            )

            # Fine tuning:
            ft_generator.fine_tune(
                score_function=score_fn,  # This should output normalised score from (0, 1)
                reference_conformer=c_task["reference_conformer"],
                variance=c_task["variance"],
                # RL Fine-tune params
                n_epochs=20,
                train_batch_size=128,
                eval_batch_size=128,
                lambda_edm_adapter=1.0,
                lambda_edm_reg=0.2,
                learning_rate=8e-5,
                sigma=60.0,
                temperature=1.5,
                n_samples_per_mol=8,
                reward_clip=(-1.0, 1.0),
                eval_every=4,
                save_dir=ft_checkpoints_dir,
            )

            # Taking the latest checkpoint by default
            print("Fine Tuning finished loading last checkpoint...")
            ft_generator.load_fine_tune_checkpoint(f"{ft_checkpoints_dir}/latest_checkpoint.pt")
            bl_generator = MLConformerGenerator(
                edm_weights="./edm_moi_chembl_15_39.pt",
                adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
                device=device,
                diffusion_steps=d_steps,
            )

            for task_id in range(N_TASKS):
                stats = evaluate(
                     bl_generator=bl_generator,
                     ft_generator=ft_generator,
                     score_fn={'name': score_name, 'function': score_fn},
                     task=c_task,
                     agent_writer=FT_SAMPLES_WRITER,
                     baseline_writer=BL_SAMPLES_WRITER,
                    )


def scoring_function(mols) -> list[float]:
    smilies = []
    invalid_mask = []
    duplicate_mask = []

    for mol in mols:


    score_results = reinvent_scoring_function

    scores = score_results.total_scores

    return None

