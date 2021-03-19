from PIL import Image
from torch.utils.data import Dataset

from .utils import list_dir, list_files

class StyleGAN2_ADA_PyTorch(Dataset):
    def __init__(self, path, transform):

        self.main_dir = path
        image_list = []

        folder_list = list_dir(self.main_dir, prefix=True)
        for folder in folder_list:
            all_files = list_files(folder, "png",prefix=True)
            image_list += all_files

        self.total_imgs = sorted(image_list)
        self.transform = transform

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, index):

        X = Image.open(self.total_imgs[index])
        
        if self.transform is not None:
            X = self.transform(X)

        target = 0 # no labels

        return X, target
