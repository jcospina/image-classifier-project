import os
import json
import torch
from PIL import Image
from torchvision import datasets, transforms

def load_data(data_dir = "./flowers"):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),    
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform = train_transforms),
        "test": datasets.ImageFolder(test_dir, transform = validation_transforms),
        "validation": datasets.ImageFolder(valid_dir, transform = validation_transforms)
    }
    
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=64, shuffle=True),
    }
    
    return image_datasets, dataloaders

def save_model(model, optimizer, image_dataset, arch, hidden_units, dropout, learning_rate, save_folder):
    model_inputs = model.classifier[0].in_features
    model.class_to_idx = image_dataset.class_to_idx
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(
        {
            "input_size": model_inputs,
            "arch": arch,
            "hidden_units": hidden_units,
            "output_size": 102,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "optimizer": optimizer.state_dict(),
            "state_dict": model.state_dict(),
            "class_to_idx": model.class_to_idx,
        },
        f"{save_folder}/checkpoint.pth",
    )

def load_model(path):
    if not os.path.exists(path):
        print("Checkpoint doesn't exist")
        return None
    checkpoint = torch.load(path)
    return checkpoint

def process_image(image_path):
    with Image.open(image_path) as im:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        processed_image = transform(im)
    return processed_image

def get_category_labels_dict(path):    
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name