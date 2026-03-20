import torch
import torch.nn.functional as F


def split_route_update_batch(train_input, train_target, route_ratio):
    if not 0.0 < route_ratio < 1.0:
        raise ValueError("route_ratio must be in the open interval (0, 1)")

    if train_input.shape[0] >= 2:
        if train_input.shape[0] != train_target.shape[0]:
            raise ValueError("train_input and train_target must have the same batch size")
        num_samples = int(train_input.shape[0])
        route_count = int(round(num_samples * route_ratio))
        route_count = max(1, min(num_samples - 1, route_count))

        permutation = torch.randperm(num_samples, device=train_input.device)
        route_indices = permutation[:route_count]
        update_indices = permutation[route_count:]

        return {
            "route_input": train_input[route_indices],
            "route_target": train_target[route_indices],
            "update_input": train_input[update_indices],
            "update_target": train_target[update_indices],
        }

    if train_input.ndim < 3:
        raise ValueError("need at least 3 dimensions when falling back to time-axis splitting")
    if train_input.shape[:2] != train_target.shape[:2]:
        raise ValueError("train_input and train_target must share the same leading dimensions")

    seq_len = int(train_input.shape[1])
    if seq_len < 2:
        raise ValueError("need at least 2 time steps to build route/update splits")

    route_count = int(round(seq_len * route_ratio))
    route_count = max(1, min(seq_len - 1, route_count))

    return {
        "route_input": train_input[:, :route_count, :],
        "route_target": train_target[:, :route_count, :],
        "update_input": train_input[:, route_count:, :],
        "update_target": train_target[:, route_count:, :],
    }


def compute_route_softmax_weights(route_losses, temperature):
    loss_tensor = torch.as_tensor(route_losses, dtype=torch.float32)
    if loss_tensor.ndim != 1:
        raise ValueError("route_losses must be a 1D tensor or sequence")
    if loss_tensor.numel() == 0:
        raise ValueError("route_losses cannot be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    return F.softmax((-loss_tensor) / float(temperature), dim=0)


def aggregate_shared_states_by_weights(shared_state_list, weights):
    if not shared_state_list:
        raise ValueError("shared_state_list cannot be empty")

    weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
    if weight_tensor.ndim != 1:
        raise ValueError("weights must be a 1D tensor or sequence")
    if weight_tensor.numel() != len(shared_state_list):
        raise ValueError("weights length must match shared_state_list length")

    total_weight = float(weight_tensor.sum().item())
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    normalized_weights = weight_tensor / total_weight

    reference_keys = list(shared_state_list[0].keys())
    aggregated_state = {}
    for name in reference_keys:
        reference_tensor = shared_state_list[0][name]
        weighted_sum = torch.zeros_like(reference_tensor)
        for idx, state_dict in enumerate(shared_state_list):
            weighted_sum = weighted_sum + state_dict[name].to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            ) * normalized_weights[idx].to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )
        aggregated_state[name] = weighted_sum
    return aggregated_state
