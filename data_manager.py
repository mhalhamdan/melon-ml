
import pandas as pd
import numpy as np
import torch
import torchvision


from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from shutil import copyfile
from PIL import Image

# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)
ROOT_DIR = "melon_imgs_dataloader"
# class_names = {'cantaloupe': 0, 'honeydew': 1, 'melon': 2, 'watermelon': 3}
class_names = ('cantaloupe', 'honeydew', 'melon', 'watermelon')

# Split = (train, val, test)
def preprocess(data, split=(0.7, 0.1, 0.2)):
    
    PHASES = ["train", "val", "test"]

    DATA_DIR = "data3"

    LABEL_MAP = {
        "cantaloupe": "melon",
        "melon": "melon",
        "watermelon": "melon",
        "honeydew": "melon",
        "people": "not_melon",
        "object": "not_melon",
        "apple": "not_melon",
        "orange": "not_melon"
    }

    errors = 0

    for label, contents in data.items():


        indices = tuple(round(i * (len(contents)-2)) for i in split)
        
        a = b = 0

        for idx, phase in enumerate(PHASES):
            try:
                os.mkdir(f"{ROOT_DIR}/{DATA_DIR}/{phase}")
                os.mkdir(f"{ROOT_DIR}/{DATA_DIR}/{phase}/{LABEL_MAP[label]}")
            except:
                pass
            
            # Set indices
            if a != 0:
                a = b+1
            
            b = indices[idx] + a - 1

            
            for file in contents[a:b]:
                new_file = is_valid_file(f"{ROOT_DIR}/{label}/{file}", file)

                if new_file:
                    copyfile(src=f"{ROOT_DIR}/{label}/{new_file}", dst=f"{ROOT_DIR}/{DATA_DIR}/{phase}/{LABEL_MAP[label]}/{new_file}")
                else:
                    errors += 1

            print(a,b)
            a = b
    
    print("total errors: ", errors)

def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg.convert("RGB")

    else:
        return im

def is_valid_file(path, filename):
    

    if path.endswith((".png", ".jpg", ".jpeg")):
        try:
            img = Image.open(path) # open the image file

            img.verify() # verify that it is, in fact an image

            # img = remove_transparency(img)# If transparent remove transparency
            # img.convert("RGB")

            if path.endswith(".png"):
                filename.replace(".png", "jpg")

            img = Image.open(path) # open the image file

            img = img.convert("RGB")

            # exif_data = img._getexif()

            img.save(path)

            return filename
        except Exception as e:
            print('Bad file:', (path, e)) # print out the names of corrupt files
            return False
    else:
        return False
    

def load_data(data_dir): 
    from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize
    from torchvision.transforms import ToTensor, Normalize

    from torch.utils.data import Subset
    from torch.utils.data import DataLoader

    from PIL.Image import BICUBIC

    batch_size = 4

    train_transform = Compose([
        Resize(256, BICUBIC),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = Compose([
        Resize(224, BICUBIC),    
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  

    data_transform = {"train": train_transform, "val": test_transform, "test": test_transform}

    # Create training and validation datasets and testing dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['train', 'val', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, d in dataloaders_dict.items():
        print(i,len(d))

    visualize(dataloaders_dict["train"])


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    
    if title is not None:
        plt.title(title)

    plt.show()    

def visualize(data_loader):

    inputs, classes = next(iter(data_loader))
    print(classes)
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
 

        



if __name__ == "__main__":

    if True:
        dirs = ["cantaloupe", "honeydew", "melon", "watermelon", "object", "people", "apple", "orange"]
        data = {}

        for name in dirs:
            contents = [f for f in os.listdir(f"{ROOT_DIR}/{name}") if os.path.isfile(f"{ROOT_DIR}/{name}/{f}")]
            data[name] = contents

        preprocess(data, (0.7, 0.2, 0.1))


    # data_dir = "./melon_imgs_dataloader/data"
    # num_classes = 2
    # batch_size = 10

    # load_data(data_dir)

