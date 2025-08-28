import os
import numpy as np
import PIL
from PIL import Image

from torch.utils.data import Dataset
from datasets import load_dataset


DATA_DIR= "data/celebahq"


class CelebAHQ_caption(Dataset):
    def __init__(self, split = 'train', split_rate=[0.8,0.1,0.1], size = None, interpolation="bicubic"):
        self.size = size
        self.interpolation = {"bilinear":Image.BILINEAR,
                              "bicubic":Image.BICUBIC,
                              "lanczos":Image.LANCZOS,
                              }[interpolation]

        # Load the datasetds from Hugging Face
        metadata = load_dataset("Ryan-sjtu/celebahq-caption", cache_dir = f"{DATA_DIR}/celebahq_caption-cache",split = 'train')
        self.metadata = self.filer_train_val_test(metadata, split, split_rate)

    def filer_train_val_test(self, metadata, split, split_rate):
        """filter metadata into train, val, test sets based on split rate"""
        assert np.isclose(np.sum(split_rate), 1), "split_rate must sum to 1"
        metadata = metadata.shuffle(seed=42)
        n = len(metadata)
        split_num_1 = int(n * split_rate[0])
        split_num_2 = int(n * (split_rate[0] + split_rate[1]))
        if split == 'train':
            return metadata.select(range(0, split_num_1))
        elif split == 'val':
            return metadata.select(range(split_num_1, split_num_2))
        elif split == 'test':
            return metadata.select(range(split_num_2, n))
        else:
            raise ValueError("split must be one of ['train', 'val', 'test']")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image = item['image']
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32) # sd set image range to [-1, 1]
        item['jpg'] = image
        item['txt'] = item['text']
        return item
    
if __name__ == '__main__':
    dataset = CelebAHQ_caption(split='train', size=256)
    print(f"Dataset length: {len(dataset)}")

    item = dataset[1234]
    jpg = item['jpg']
    txt = item['txt']
    print(f"Text: {txt}")
    print(f"Image size: {jpg.shape}")  # PIL Image size
    print(txt)