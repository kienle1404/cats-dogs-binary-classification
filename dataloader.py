# Dataset
# Dataloader & collate function
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from pathlib import Path
from glob import glob
import os
import cv2 # BGR
import imageio
from PIL import Image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform) -> None:
        super().__init__()
        
    
        self.class_name = [os.path.basename(i) for i in glob(os.path.join(data_dir, "*"))]
        self.class_id = {i : j for j, i in enumerate(self.class_name)}
        self.image_list = glob(os.path.join(data_dir, "*", "*.jpg"))
        
        self.images = []
        self.labels = []
        for image_path in tqdm(self.image_list): 
            # training_set/cats/cat1.jpg -> self.labels = [0]
            self.labels.append(self.class_id[os.path.split(image_path)[-2]])
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = imageio.mimread(image_path)
            img = torch.from_numpy(img) /255.0 # [H, W, 3]
            img = transform(img)
            self.images.append(img)
            
        self.images = self.images[:10]
        self.labels = self.labels[:10]
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    

def get_dataset(cfg: OmegaConf): # config.dataset
    IMAGE_SIZE=(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)

    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()])

    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()])

    # Turn image folders into Datasets
    # train_data_augmented = datasets.ImageFolder(cfg.train_dir, transform=train_transform)
    # test_data_augmented = datasets.ImageFolder(cfg.test_dir, transform=test_transform)
    train_data_augmented = ImageDataset(cfg.train_dir, transform=train_transform)
    test_data_augmented = ImageDataset(cfg.test_dir, transform=train_transform)
    return train_data_augmented, test_data_augmented


# input: image; label: 0; 1
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform) -> None:
        super().__init__()
        
    
        self.class_name = [os.path.basename(i) for i in glob(os.path.join(data_dir, "*"))]
        self.class_id = {i : j for j, i in enumerate(self.class_name)}
        self.image_list = glob(os.path.join(data_dir, "*", "*.jpg"))[:100]
        torch.save(self.class_id, 'experiments\\mapping.pt')
        self.images = []
        self.labels = []
        for image_path in tqdm(self.image_list): 
            # training_set/cats/cat1.jpg -> self.labels = [0]
            self.labels.append(self.class_id[image_path.split("\\")[-2]])
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = imageio.mimread(image_path)
            img = Image.fromarray(img)
            img = transform(img)
            self.images.append(img)
            
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]



class LoadDataLoader():
    def __init__(self, cfg: OmegaConf):
        self.config_train = cfg.dataloader.train
        print(self.config_train)
        self.config_test = cfg.dataloader.test
        self.train_dataset, self.test_dataset = get_dataset(cfg)
    
    def get_train_dataloader(self):
        train_dataloader_augmented = DataLoader(self.train_dataset, 
                                            batch_size=self.config_train.batchsize, 
                                            shuffle=self.config_train.shuffle,
                                            num_workers=self.config_train.num_worker)
        
        return train_dataloader_augmented

    def get_test_dataloader(self):
        test_dataloader_augmented = DataLoader(self.test_dataset, 
                                            batch_size=self.config_test.batchsize, 
                                            shuffle=self.config_test.shuffle, 
                                            num_workers=self.config_test.num_worker)
        return test_dataloader_augmented