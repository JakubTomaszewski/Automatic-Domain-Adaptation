import torch


def get_available_devices() -> list:
    """Returns all available devices for computation.

    Returns:
        list: available device for computation
    """
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')
    print('Available devices:', devices)
    return devices


def available_torch_device(device):
    """Returns the device available for computation.

    Returns:
        torch.device: device available for computation
    """
    if device in get_available_devices():
        print(f'Chosen device: {device}')
        return torch.device(device)
    else:
        return torch.device('cpu')
