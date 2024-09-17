import torch
import tqdm

from ..typing import Device, Literal, Module, Tensor, Tuple

__all__ = [
    "optimize_geodesic",
]


# def geodesic_jacobian(
#     model: Module,
#     path: Tensor,
#     steps: int,
#     input_dim: int,
# ) -> Tensor:
#     # Compute Jacobians for all points along the path in a batched way
#     # We pass the entire path and compute the Jacobian for all points simultaneously
#     jacobians: Tensor = torch.autograd.functional.jacobian(
#         model, path
#     )  # Shape: (steps, *output_dim, steps, *input_dim)

#     # Reshape the Jacobians to flatten the input/output dimensions
#     jacobians_flat = jacobians.reshape(
#         steps, -1, steps, input_dim
#     )  # Shape: (steps, output_dim, steps, input_dim)

#     # Extract diagonals only,
#     # which corresponds to the relationship between each input and its own output
#     jacobians_flat = torch.cat(
#         [jacobians_flat[i, :, i, :] for i in range(steps)], dim=0
#     )  # Shape: (steps, output_dim, input_dim)

#     return jacobians_flat


def geodesic_jacobian(
    model: Module,
    path: Tensor,
    steps: int,
    input_dim: int,
) -> Tensor:
    # Changing computation for memory efficiency
    jacobians_flat = []

    for i in range(steps):
        # Make sure each input point has requires_grad=True
        path[i].requires_grad_(True)

        jacobian_i: Tensor = torch.autograd.functional.jacobian(
            model, path[i].unsqueeze(0)
        )  # Shape: (1, *output_dim, *input_dim)

        # Flatten the input/output dimensions
        jacobian_i_flat = jacobian_i.reshape(
            1, -1, input_dim
        )  # Shape: (1, output_dim, input_dim)

        # Use cpu for storing jacobians, later move to working devices
        jacobians_flat.append(jacobian_i_flat.detach().to(torch.device("cpu")))

    jacobians_flat = torch.cat(
        jacobians_flat, dim=0
    )  # Shape: (steps, output_dim, input_dim)

    return jacobians_flat.to(path.device)


def compute_geodesic_energy_gradient(
    model: Module,
    path: Tensor,
    delta_t: float,
) -> Tensor:
    steps = path.shape[0]
    input_dim = path[0].numel()

    # Compute g(z_{i+1}), g(z_i), g(z_{i-1}) in batches
    g_path = model(path)  # Compute g(z_i) for the entire path in one batch

    # Compute forward and backward differences for all intermediate points in one go
    forward_diff = (
        g_path[2:] - g_path[1:-1]
    )  # g(z_{i+1}) - g(z_i), Shape: (steps-2, *output_dim)
    backward_diff = (
        g_path[1:-1] - g_path[:-2]
    )  # g(z_i) - g(z_{i-1}), Shape: (steps-2, *output_dim)

    # Compute the acceleration term in batch: g(z_{i+1}) - 2 * g(z_i) + g(z_{i-1})
    acceleration = forward_diff - backward_diff  # Shape: (steps-2, *output_dim)

    # Reshape the acceleration for batch-wise computation
    acceleration_flat = acceleration.view(steps - 2, -1)  # Shape: (steps-2, output_dim)

    # Compute Jacobians for all intermediate points in one batch
    jacobians_flat = geodesic_jacobian(
        model, path[1:-1], steps - 2, input_dim
    )  # Shape: (steps-2, output_dim, input_dim)

    # Compute the gradient for all intermediate points in a batched manner
    # ∇ziE = - 1/δt * Jg^T(zi) * (g(z_{i+1}) - 2g(zi) + g(z_{i-1}))
    gradients_flat = -(1 / delta_t) * torch.einsum(
        "bji,bj->bi", jacobians_flat, acceleration_flat
    )

    # Reshape the gradients to match the input shape of path
    gradients = gradients_flat.view(
        steps - 2, *path.shape[1:]
    )  # Shape: (steps-2, *input_shape)

    return gradients


def compute_geodesic_length(
    model: Module,
    path: Tensor,
) -> Tensor:
    steps = path.shape[0]
    input_dim = path[0].numel()

    # Compute differences between successive points in the path (dx)
    dx = path[1:] - path[:-1]  # Shape: (steps-1, *input_dim)
    dx_flat = dx.reshape(
        steps - 1, input_dim
    )  # Flatten each dx to shape (steps-1, input_dim)

    # Compute Jaccobian
    jacobians_flat = geodesic_jacobian(model, path, steps, input_dim)

    # Slice out the last Jacobian to match the steps-1 of dx
    jacobians_flat = jacobians_flat[:-1]  # Shape: (steps-1, feature_dim, input_dim)

    # Now compute the riemannian metric, i.e., J^T * J
    # einsum('bji,bjk->bik') computes J^T * J
    metric = torch.einsum("bji,bjk->bik", jacobians_flat, jacobians_flat)

    # Compute the geodesic length
    # einsum('bi,bik,bk->b') computes (dx^T * JTJ * dx) for each step
    ds_squared = torch.einsum("bi,bik,bk->b", dx_flat, metric, dx_flat)

    # Compute the total length by summing over all the steps
    total_length = torch.sum(torch.sqrt(ds_squared))

    return total_length


def optimize_geodesic_with_energy(
    model: Module,
    x1: Tensor,
    x2: Tensor,
    steps: int = 100,
    lr: float = 0.01,
    iterations: int = 100,
    device: Device = "cpu",
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    model.eval().to(device)
    x1 = x1.to(device)
    x2 = x2.to(device)

    delta_t = 1 / steps  # Time step size (δt)

    # Initialize the path as a line between x1 and x2 for intermediate points only
    t_values = torch.linspace(0, 1, steps, device=device)[1:-1]
    path_intermediate = torch.stack([x1 * (1 - t) + x2 * t for t in t_values], dim=0)

    # Make sure intermediate points are leaf tensors and require gradients
    path_intermediate = path_intermediate.clone().detach().requires_grad_(True)

    # Create optimizer only for the intermediate points
    optimizer = torch.optim.Adam([path_intermediate], lr=lr)

    energy_gradients_l2 = []

    for _ in tqdm.trange(
        iterations,
        desc="Geodesic optimization using energy gradient",
        disable=not verbose,
    ):
        optimizer.zero_grad()

        # Concatenate x1, path_intermediate, and x2 to form the full path
        path = torch.cat([x1.unsqueeze(0), path_intermediate, x2.unsqueeze(0)], dim=0)

        # Compute the energy gradient
        energy_gradients = compute_geodesic_energy_gradient(model, path, delta_t)

        # Manually set gradients for path_intermediate (z_1 to z_{T-1})
        path_intermediate.grad = energy_gradients.detach()

        # Perform optimization step
        optimizer.step()

        # Track l2 norm of energy gradients
        energy_gradients_l2.append(
            energy_gradients.reshape(steps - 2, -1)
            .norm(dim=-1)
            .detach()
            .to(torch.device("cpu"))
            .unsqueeze(dim=0)
        )

        torch.cuda.empty_cache()  # Free unused memory

    # Return the full path including x1 and x2 & l2 norm of energy gradients
    geodesic = (
        torch.cat([x1.unsqueeze(0), path_intermediate, x2.unsqueeze(0)], dim=0)
        .detach()
        .to(torch.device("cpu"))
    )
    energy_gradients_l2 = torch.cat(energy_gradients_l2, dim=0).to(torch.device("cpu"))

    return geodesic, energy_gradients_l2


def optimize_geodesic_with_length(
    model: Module,
    x1: Tensor,
    x2: Tensor,
    steps: int = 100,
    lr: float = 0.01,
    iterations: int = 100,
    device: Device = "cpu",
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    model.eval().to(device)
    x1 = x1.to(device)
    x2 = x2.to(device)

    # Initialize the path as a line between x1 and x2 for intermediate points only
    t_values = torch.linspace(0, 1, steps, device=device)[1:-1]
    path_intermediate = torch.stack([x1 * (1 - t) + x2 * t for t in t_values], dim=0)

    # Make sure intermediate points are leaf tensors and require gradients
    path_intermediate = path_intermediate.clone().detach().requires_grad_(True)

    # Create optimizer only for the intermediate points
    optimizer = torch.optim.Adam([path_intermediate], lr=lr)

    lengths = []

    for _ in tqdm.trange(
        iterations,
        desc="Geodesic optimization using path length",
        disable=not verbose,
    ):
        optimizer.zero_grad()

        # Concatenate x1, path_intermediate, and x2 to form the full path
        path = torch.cat([x1.unsqueeze(0), path_intermediate, x2.unsqueeze(0)], dim=0)

        # Compute the geodesic length
        total_length = compute_geodesic_length(model, path)

        # Minimize the geodesic length
        total_length.backward()
        optimizer.step()

        # Track geodesic lengths
        lengths.append(total_length.detach().unsqueeze(dim=0))

        torch.cuda.empty_cache()  # Free unused memory

    # Return the full path including x1 and x2 & lengths
    geodesic = (
        torch.cat([x1.unsqueeze(0), path_intermediate, x2.unsqueeze(0)], dim=0)
        .detach()
        .to(torch.device("cpu"))
    )
    lengths = torch.cat(lengths, dim=0).to(torch.device("cpu"))

    return geodesic, lengths


def optimize_geodesic(
    model: Module,
    x1: Tensor,
    x2: Tensor,
    strategy: Literal[
        "energy",
        "length",
    ] = "energy",
    steps: int = 100,
    lr: float = 0.01,
    iterations: int = 100,
    device: Device = "cpu",
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    if strategy == "energy":
        return optimize_geodesic_with_energy(
            model=model,
            x1=x1,
            x2=x2,
            steps=steps,
            lr=lr,
            iterations=iterations,
            device=device,
            verbose=verbose,
        )
    elif strategy == "length":
        return optimize_geodesic_with_length(
            model=model,
            x1=x1,
            x2=x2,
            steps=steps,
            lr=lr,
            iterations=iterations,
            device=device,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
