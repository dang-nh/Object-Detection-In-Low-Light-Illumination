import torch

from data.data import ExDarkDataset, get_transforms


def collate_fn(batch):
    return tuple(zip(*batch))


def build_loaders(dataframe, image_dir, batch_size, num_workers, mode):
    transforms = get_transforms(mode)
    dataset = ExDarkDataset(dataframe, image_dir, transforms=transforms)
                            # transforms=transforms.Compose([
                            #     transforms.Resize((224, 224)),
                            #     transforms.RandomRotation(10),
                            #     transforms.RandomHorizontalFlip(),
                            #     transforms.Normalize(
                            #         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            # ]))

    if mode=="train":
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader
