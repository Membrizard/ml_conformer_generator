import time
import os
from tqdm import tqdm
import numpy as np
import torch
import copy
from models import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)

from utils import CONTEXT_NORMS, DistributionNodes, generate_evaluation_samples


class EMA:
    """
    Exponential moving Average weight decay Decay class
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Queue:
    """
    Queue for Gradient Clipping
    """

    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def prepare_context(
    node_mask: torch.Tensor, context: torch.Tensor, context_norms: dict = CONTEXT_NORMS
):
    """
    Normalises the context, using pre-computed property norms dict
    :param node_mask:
    :param context:
    :param context_norms:
    :return:
    """
    batch_size, n_nodes, _ = node_mask.size()

    properties = (context - context_norms["mean"]) / context_norms["mad"]
    reshaped = properties.unsqueeze(1).repeat(1, n_nodes, 1)
    context = reshaped * node_mask

    return context


def gradient_clipping(flow, gradnorm_queue):
    """
    Clips Gradient
    :param flow:
    :param gradnorm_queue:
    :return:
    """
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0
    )

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(
            f"Clipped gradient with value {grad_norm:.1f} "
            f"while allowed {max_grad_norm:.1f}"
        )
    return None


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def compute_loss_and_nll(
    generative_model, nodes_dist: DistributionNodes, x, h, node_mask, edge_mask, context
):
    bs, n_nodes, n_dims = x.size()

    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    nll = generative_model(x, h, node_mask, edge_mask, context)

    n = node_mask.squeeze(2).sum(1).long()

    log_pN = nodes_dist.log_prob(n)

    assert nll.size() == log_pN.size()
    nll = nll - log_pN

    # Average over batch.
    nll = nll.mean(0)

    return nll


def train_epoch(
    loader,
    model,
    model_dp,
    model_ema,
    ema,
    ema_decay: float,
    device,
    dtype,
    optim,
    nodes_dist,
    gradnorm_queue,
    context_norms: dict,
):
    model_dp.train()
    model.train()
    nll_epoch = []

    for batch in tqdm(loader):
        x = batch["x"].to(device, dtype)
        h = batch["h"].to(device, dtype)
        node_mask = batch["node_mask"].to(device, dtype).unsqueeze(2)
        edge_mask = batch["edge_mask"].to(device, dtype)
        context = batch["context"].to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        check_mask_correct([x, h], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        context = prepare_context(node_mask, context, context_norms)

        optim.zero_grad()

        # transform batch through flow
        loss = compute_loss_and_nll(
            model_dp, nodes_dist, x, h, node_mask, edge_mask, context
        )
        nll_epoch.append(loss.item())
        # standard nll from forward KL
        # loss = nll + args.ode_regularization * reg_term
        loss.backward()

        # Gradient Clipping is enabled by default
        gradient_clipping(model, gradnorm_queue)

        optim.step()

        # Update EMA if enabled.
        # EMA is enabled by default and ema_decay is 0.999
        if ema_decay > 0:
            ema.update_model_average(model_ema, model)

    return np.mean(nll_epoch)


def test(
    loader,
    eval_model,
    device,
    dtype,
    nodes_dist,
    context_norms,
):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        for batch in tqdm(loader):
            x = batch["x"].to(device, dtype)
            h = batch["h"].to(device, dtype)
            node_mask = batch["node_mask"].to(device, dtype).unsqueeze(2)
            edge_mask = batch["edge_mask"].to(device, dtype)
            context = batch["context"].to(device, dtype)

            batch_size = x.size(0)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            context = prepare_context(node_mask, context, context_norms).to(
                device, dtype
            )

            # transform batch through flow
            nll = compute_loss_and_nll(
                eval_model, nodes_dist, x, h, node_mask, edge_mask, context
            )
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size

    return nll_epoch / n_samples


def train_edm_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    nodes_dist: DistributionNodes,
    context_norms: dict = CONTEXT_NORMS,
    dtype: torch.dtype = torch.float32,
    data_parallelisation: bool = True,
    ema_decay: float = 0.999,
    epochs: int = 3000,
    report_epochs: int = 100,
    device: torch.device = "cpu",
    learning_rate: float = 1e-4,
    save: bool = True,
    generate_samples: bool = False,
    evaluation_samples: dict = None,
    save_path: str = "./edm_training_log",
    optimiser_path: str = None,
):
    os.makedirs(save_path, exist_ok=True)
    log = open(f"{save_path}/train_log.txt", "a+")

    if generate_samples:
        #  Save reference molecules to training log folder, if sample generation is enabled

        if evaluation_samples:
            eval_samples_path = f"{save_path}/evaluation_samples_xyz"
            os.makedirs(eval_samples_path, exist_ok=True)
            for i, item in enumerate(evaluation_samples["xyz_blocks"]):
                cid = evaluation_samples["ids"][i]
                with open(f"{eval_samples_path}/sample_{i+1}_{cid}.xyz", "w+") as f:
                    f.write(evaluation_samples["xyz_blocks"][i])
                with open(f"{save_path}/evaluation_samples.smi", "a+") as f:
                    f.write(f"{i+1}\t{cid}\t{evaluation_samples['smiles'][i]}")
        else:
            raise ValueError(
                "A number of Molecules must be provided as evaluation samples, if generation is enabled."
            )

    # Gradient Clipping
    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add a large number to be flushed

    model.to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=1e-12
    )

    if optimiser_path:
        print("Loading Optimizer State")
        optim.load_state_dict(
            torch.load(
                optimiser_path,
                map_location=device,
            )
        )

    # Initialize dataparallel if enabled and possible.
    if data_parallelisation and torch.cuda.device_count() > 1:
        print(f"Training using {torch.cuda.device_count()} GPUs")
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

        # Initialize model copy for exponential moving average of params.
    if ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = EMA(ema_decay)

        if data_parallelisation and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8

    context_norms = {
        key: context_norms[key].to(device, dtype) for key in context_norms.keys()
    }

    for epoch in range(epochs):
        current_epoch = epoch + 1
        print(f"EPOCH {current_epoch} out of {epochs}")

        start_epoch = time.time()

        print("Training")
        train_loss = train_epoch(
            loader=train_loader,
            model=model,
            model_dp=model_dp,
            model_ema=model_ema,
            ema=ema,
            ema_decay=ema_decay,
            device=device,
            dtype=dtype,
            nodes_dist=nodes_dist,
            context_norms=context_norms,
            gradnorm_queue=gradnorm_queue,
            optim=optim,
        )

        epoch_time = time.time() - start_epoch

        main_status = (
            f"EPOCH {current_epoch}"
            f" NLL LOSS - TRAINING = {train_loss}"
            f" LEARNING RATE = {learning_rate}"
            f" TRAINING TIME - {epoch_time:.1f} sec\n"
            "--------------------------------------------------------------------\n"
        )

        log.write(main_status)
        print(main_status)

        if epoch % report_epochs == 0:
            print("Validating")
            nll_val = test(
                loader=val_loader,
                eval_model=model_ema_dp,
                device=device,
                dtype=dtype,
                nodes_dist=nodes_dist,
                context_norms=context_norms,
            )
            print("Testing")
            nll_test = test(
                loader=test_loader,
                eval_model=model_ema_dp,
                device=device,
                dtype=dtype,
                nodes_dist=nodes_dist,
                context_norms=context_norms,
            )

            if (nll_val < best_nll_val) and save:
                # Prepare Folders
                epoch_folder = f"{save_path}/epoch_{current_epoch}"
                os.makedirs(epoch_folder, exist_ok=True)
                os.makedirs(f"{epoch_folder}/weights", exist_ok=True)
                os.makedirs(f"{epoch_folder}/optimizer", exist_ok=True)

                best_nll_val = nll_val
                best_nll_test = nll_test
                torch.save(
                    optim.state_dict(),
                    f"{epoch_folder}/optimizer/OPTIMIZER_{current_epoch}.statedict",
                )
                torch.save(
                    model.state_dict(),
                    f"{epoch_folder}/weights/EDM_MODEL_{current_epoch}.weights",
                )
                if ema_decay > 0:
                    torch.save(
                        model.state_dict(),
                        f"{epoch_folder}/weights/EDM_MODEL_EMA_{current_epoch}.weights",
                    )

                # Generate Samples and Evaluate the Similarities

                detailed_status = (
                    f"\nAdditional details at EPOCH {current_epoch}:\n"
                    f" NLL LOSS - VALIDATION = {round(nll_val, 3)}\n"
                    f" NLL LOSS - TEST = {round(nll_test, 3)}\n"
                    f" BEST NLL LOSS - VALIDATION = {round(best_nll_val, 3)}\n"
                    f" BEST NLL LOSS - TEST = {round(best_nll_test, 3)}\n\n"
                    "--------------------------------------------------------------------\n"
                )
                log.write(detailed_status)
                print(detailed_status)

                if generate_samples:
                    print("Generating Samples")

                    gen_path = f"{epoch_folder}/outputs"
                    os.makedirs(gen_path, exist_ok=True)
                    gen_start = time.time()
                    generate_evaluation_samples(
                        reference_contexts=evaluation_samples["context"],
                        generative_model=model,
                        device=device,
                        save_path=gen_path,
                    )
                    gen_time = time.time() - gen_start
                    gen_status = (
                        f"Evaluation Samples generated in {gen_time:.1f} sec\n"
                        "--------------------------------------------------------------------\n"
                    )
                    log.write(gen_status)
                    print(gen_status)

    return None
