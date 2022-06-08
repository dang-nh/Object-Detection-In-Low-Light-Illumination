from data.data import Dataset, get_transforms
import torch

def build_loaders(dataset_name, batch_size, num_workers, mode):
    transforms = get_transforms(mode)
    dataset = ExDarkDataset(
        anno_path=dataset_name,
        transforms = transforms
        
    )
    
    if mode == "train":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )