import torch
import random


def clone_state_dict(state_dict):
    cloned_state = {}
    for name, tensor in state_dict.items():
        cloned_state[name] = tensor.detach().clone()
    return cloned_state


def update_best_meta_checkpoint(best_record, epoch_index, query_loss, shared_state):
    query_loss = float(query_loss)
    if best_record is not None and query_loss >= best_record["query_loss"]:
        return best_record

    return {
        "epoch": int(epoch_index),
        "query_loss": query_loss,
        "shared_state": clone_state_dict(shared_state),
    }


def compose_personalized_meta_init_state(shared_state, local_state):
    mixed_state = clone_state_dict(shared_state)
    mixed_state.update(clone_state_dict(local_state))
    return mixed_state


def accumulate_weighted_state(state_sum, state_dict, weight):
    weight = float(weight)
    if weight <= 0:
        raise ValueError("weight must be positive")

    if state_sum is None:
        return {
            name: tensor.detach().clone() * weight
            for name, tensor in state_dict.items()
        }

    for name, tensor in state_dict.items():
        state_sum[name].add_(tensor.detach(), alpha=weight)
    return state_sum


def finalize_weighted_state(state_sum, total_weight):
    total_weight = float(total_weight)
    if state_sum is None:
        raise ValueError("state_sum cannot be None")
    if total_weight <= 0:
        raise ValueError("total_weight must be positive")

    return {
        name: tensor / total_weight
        for name, tensor in state_sum.items()
    }


def select_meta_shared_state(best_record, latest_shared_state, save_best_only):
    if save_best_only and best_record is not None:
        return clone_state_dict(best_record["shared_state"])
    return clone_state_dict(latest_shared_state)


def should_stop_early(no_improve_rounds, patience):
    return patience > 0 and no_improve_rounds >= patience


def choose_local_meta_task_classes(class_sample_counts, n_support, n_query, task_mode="cross_cluster", rng=None):
    rng = random if rng is None else rng
    if task_mode not in {"cross_cluster", "same_cluster"}:
        raise ValueError(f"unsupported task_mode: {task_mode}")

    if n_support <= 0 or n_query <= 0:
        raise ValueError("n_support and n_query must be positive")

    same_cluster_candidates = [
        class_idx
        for class_idx, sample_count in enumerate(class_sample_counts)
        if sample_count >= (n_support + n_query)
    ]

    if task_mode == "same_cluster":
        if not same_cluster_candidates:
            raise ValueError("no class has enough samples for same-cluster support/query sampling")
        chosen_class = rng.choice(same_cluster_candidates)
        return chosen_class, chosen_class

    support_candidates = [
        class_idx
        for class_idx, sample_count in enumerate(class_sample_counts)
        if sample_count >= n_support
    ]
    query_candidates = [
        class_idx
        for class_idx, sample_count in enumerate(class_sample_counts)
        if sample_count >= n_query
    ]
    cross_cluster_candidates = [
        (support_class_idx, query_class_idx)
        for support_class_idx in support_candidates
        for query_class_idx in query_candidates
        if support_class_idx != query_class_idx
    ]

    if cross_cluster_candidates:
        return rng.choice(cross_cluster_candidates)

    if same_cluster_candidates:
        chosen_class = rng.choice(same_cluster_candidates)
        return chosen_class, chosen_class

    raise ValueError("no class combination has enough samples for local meta task sampling")
