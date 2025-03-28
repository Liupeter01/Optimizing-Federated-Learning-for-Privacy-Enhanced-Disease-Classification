import torch
import numpy as np

# Noise function
def add_gaussian_noise(state_dict, epsilon=1.0, delta=1e-5, sensitivity=1.0):
    noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    for key in state_dict:
        if torch.is_floating_point(state_dict[key]):
            noise = torch.normal(0, noise_scale, size=state_dict[key].shape, device=state_dict[key].device)
            noise = noise.to(state_dict[key].dtype)
            state_dict[key] += noise
    return state_dict

def add_laplace_noise(state_dict, epsilon=1.0, sensitivity=1.0):
    noise_scale = sensitivity / epsilon

    for key in state_dict:
        if torch.is_floating_point(state_dict[key]):
            noise = torch.tensor(np.random.laplace(0, noise_scale, state_dict[key].shape), device=state_dict[key].device, dtype=state_dict[key].dtype)
            state_dict[key] += noise
    return state_dict

def add_adaptive_noise(state_dict, epsilon=1.0, delta=1e-5):
    for key, param in state_dict.items():
        if torch.is_floating_point(param):
            sensitivity = torch.norm(param)
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            noise = torch.normal(0, noise_scale, size=param.shape, device=param.device, dtype=param.dtype)
            param += noise
    return state_dict

def clip_gradients(state_dict, max_norm=1.0, device=None):
    # Ensure device is initialized
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move all tensors to the same device
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device)

    # Perform norm calculation
    total_norm = torch.sqrt(sum(
        torch.norm(param)**2 for param in state_dict.values() if torch.is_floating_point(param)
    ))

    scale = max_norm / (total_norm + 1e-6)
    if scale < 1:
        for key in state_dict:
            if torch.is_floating_point(state_dict[key]):
                state_dict[key] *= scale
    return state_dict

def apply_dp_to_pth(pth_path, output_path, epsilon=1.0, delta=1e-5, max_norm=1.0):
    # load model .pth
    state_dict = torch.load(pth_path)

    # Convert to a single flattened vector
    vector = torch.cat([param.flatten() for param in state_dict.values() if torch.is_floating_point(param)])
    
    # Print the vector and its size
    print("Initial Vector:", vector)
    print("Vector Size:", vector.shape[0])

    # Apply clipping and DP
    state_dict = clip_gradients(state_dict, max_norm)
    #state_dict = add_gaussian_noise(state_dict, epsilon, delta)
    #state_dict = add_laplace_noise(state_dict, epsilon)
    state_dict = add_adaptive_noise(state_dict, epsilon, delta)

    # Convert to a single flattened vector
    vector = torch.cat([param.flatten() for param in state_dict.values() if torch.is_floating_point(param)])
    
    # Print the vector and its size
    print("Vector After DP:", vector)
    print("Vector Size After DP:", vector.shape[0])

    # Save modified model
    torch.save(state_dict, output_path)
    print(f"Applied DP and saved to {output_path}")
