import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset, load_from_disk

# 配置参数
DATA_DIR= "data/laionart"
CACHE_DIR = f"{DATA_DIR}/laion-art-cache"
SAVE_DIR = f"{DATA_DIR}/images"
assert os.path.exists(DATA_DIR), f"Data directory {DATA_DIR} does not exist."
assert os.path.exists(SAVE_DIR), f"Image directory {SAVE_DIR} does not exist."

class LaionArt_Base(Dataset):
    def __init__(self, lang = ['en'], top = 0, bottom = 1_000_000, split = 'train', split_rate=[0.9,0.05,0.05], warning_threshold=0.8, size = None, interpolation="bicubic"):
        self.lang = lang
        self.top = top
        self.bottom = bottom
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        metadata = self.get_metadata()
        metadata = self.filer_train_val_test(metadata, split, split_rate)
        self.metadata = self.filter_exist(metadata, warning_threshold)

    # 加载或生成元数据
    def get_metadata(self):
        LANGUAGES = self.lang
        START_IDX = self.top
        END_IDX = self.bottom
        SUBSET_PATH = f"{DATA_DIR}/subset_metadata/top_{LANGUAGES}_{START_IDX}_{END_IDX}"
        if os.path.exists(SUBSET_PATH):
            print(f"Loading subset from disk: {SUBSET_PATH}")
            subset = load_from_disk(SUBSET_PATH)
            print(f"Loaded {len(subset)} items.")
        else:
            print("Loading laion-art dataset and generating subset...")
            ds = load_dataset("laion/laion-art", cache_dir=CACHE_DIR)
            print(f"Filtering languages: {LANGUAGES}")
            filtered = ds['train'].filter(lambda x: x['LANGUAGE'] in LANGUAGES)
            print(f"Selecting from {START_IDX} to {END_IDX} (total {END_IDX-START_IDX})...")
            subset = filtered.sort('aesthetic', reverse=True).select(range(START_IDX, END_IDX))
            print(f"Selected {len(subset)} items. Saving subset to disk...")
            subset.save_to_disk(SUBSET_PATH)
        return subset
    
    def filer_train_val_test(self, metadata, split, split_rate):
        """filter metadata into train, val, test sets based on split rate"""
        assert np.sum(split_rate) == 1, "split_rate must sum to 1"
        split_num_1 = int(len(metadata) * split_rate[0])
        split_num_2 = int(len(metadata) * (split_rate[0] + split_rate[1]))

        import random
        random.seed(42)  # for reproducibility
        random.shuffle(metadata)
        if split == 'train':
            return metadata[:split_num_1]
        elif split == 'val':
            return metadata[split_num_1:split_num_2]
        elif split == 'test':
            return metadata[split_num_2:]
        else:
            raise ValueError("split must be one of ['train', 'val', 'test']")
    
    def filter_exist(self, metadata, warning_threshold):
        """filter metadata based on existing images"""
        len_expected = len(metadata)
        existing_files = set(os.listdir(SAVE_DIR))
        for i in range(len(metadata) - 1, -1, -1):
            img_hash = metadata[i]['hash']
            filename = f"{img_hash}.jpg"
            if filename not in existing_files:
                del metadata[i]
        print(f"Filtered metadata, remaining items: {len(metadata)}")
        len_real = len(metadata)
        assert len_real >= len_expected * warning_threshold, f"Filtered metadata is too small: {len_real} < {len_expected} * {warning_threshold}. Either download images or adjust your expectation."
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_hash = item['hash']
        img_path = os.path.join(self.image_dir, f"{img_hash}.jpg")

        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32) # sd set image range to [-1, 1]

        item_out = dict(item)
        item_out['jpg'] = image
        item_out['txt'] = item_out['TEXT']
        return item_out


if __name__=='__main__':
    a = LaionArt_Base()
    b = a.__getitem__(0)["jpg"]
    b = (b+1.0)/2*225
    b = b[:,:,].astype(np.uint8)
    print(b.shape)
    b = Image.fromarray(b)
    b = a.__getitem__(0)["txt"]
    print(b)
