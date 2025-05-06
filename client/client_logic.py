import random
import ml_vector_pb2
from resnet18 import ml_resnet18, ml_modeltraining
from opacus.validators import ModuleValidator
from dp import diffprivacy
import torch
import ml_vector_pb2_grpc
# import queue

def save_fednorm_json(model, save_path):
    model_dict = model.state_dict()
    json_data = {}

    l2_terms = [v.norm(2) ** 2 for v in model_dict.values()
                if torch.is_floating_point(v)]
    total_norm = torch.sqrt(torch.sum(torch.stack(l2_terms)))

    for k, v in model_dict.items():
        if torch.is_floating_point(v):
            val = (v / total_norm).detach().cpu()
            json_data[k] = val.tolist() if val.dim() > 0 else [val.item()]

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump({
            "round": 0,
            "normalized_weights": json_data,
            "norm_value": total_norm.item()
        }, f, indent=2)


def generate_initial_model(group_name="Group1_iid", config=None):
    return ml_resnet18.resnet18(
        f"./{group_name}_labels.csv",
        f"./output/{group_name}",
        f"./train_{group_name}_smart.csv",
        f"./best_resnet18_{group_name}_custom_dp.pth",
        config
    )

def get_local_vector(group_name, config):
    return generate_initial_model(group_name, config)

def handle_merged_vector(merged_vector, local_vector, config):
    model = ml_modeltraining.create_model(**config)
    model = ModuleValidator.fix(model)

    # 加载 local_vector
    state_dict = model.state_dict()
    ptr = 0
    for key in state_dict:
        if torch.is_floating_point(state_dict[key]):
            numel = state_dict[key].numel()
            state_dict[key] = torch.tensor(local_vector[ptr:ptr + numel]).view(state_dict[key].shape)
            ptr += numel
    model.load_state_dict(state_dict)

    # 再加载 merged_vector 进行融合
    ptr = 0
    for key in model.state_dict():
        if torch.is_floating_point(model.state_dict()[key]):
            numel = model.state_dict()[key].numel()
            merged_tensor = torch.tensor(merged_vector[ptr:ptr + numel]).view(model.state_dict()[key].shape)
            model.state_dict()[key] += merged_tensor
            model.state_dict()[key] /= 2
            ptr += numel

    # 导出新的 local_vector
    merged_vec = []
    for v in model.state_dict().values():
        if torch.is_floating_point(v):
            merged_vec.extend(v.flatten().tolist())

    print(f"[Client] ✅ Local vector merged with server model. First 10 values: {merged_vec[:10]}")
    return merged_vec
