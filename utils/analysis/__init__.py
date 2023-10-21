import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def collect_feature(data_loader: DataLoader, feature_extractor1: nn.Module,feature_extractor2,
                                   device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features), :math:`|\mathcal{F}|`).
    """
    feature_extractor1.eval()
    feature_extractor2.eval()
    all_features = []
    labels=[]
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            feature = feature_extractor1(images)
            feature=feature_extractor2(feature).cpu()
            all_features.append(feature)
            labels.append(target)
            if max_num_features is not None and i >= max_num_features:
                break
    return torch.cat(all_features, dim=0),torch.cat(labels, dim=0)