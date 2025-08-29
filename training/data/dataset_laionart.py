import os
import numpy as np
import PIL
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset, load_from_disk

from .caption import caption_canny, caption_mlsd, caption_scribble, caption_softedge
from .caption import caption_depth, caption_normal, caption_seg
from .caption import caption_openpose
from .caption import caption_linear, caption_lineartanime
from .caption import caption_shuffle, caption_inpaint
from .caption import clean_models 

# 配置参数
DATA_DIR= "data/laionart"
CACHE_DIR = f"{DATA_DIR}/laion-art-cache"
SAVE_DIR = f"{DATA_DIR}/images"
CAPTION_DIR = f"{DATA_DIR}/captions"
assert os.path.exists(DATA_DIR), f"Data directory {DATA_DIR} does not exist."
assert os.path.exists(SAVE_DIR), f"Image directory {SAVE_DIR} does not exist."

class LaionArt_Base(Dataset):
    def __init__(self, lang = ['en'], top = 0, bottom = 1_000_000, split = 'train', split_rate=[0.9,0.05,0.05], warning_threshold=0.8, size = 512, interpolation="bicubic"):
        self.lang = lang
        self.top = top
        self.bottom = bottom
        self.size = size
        self.interpolation = {"bilinear":Image.BILINEAR,
                              "bicubic":Image.BICUBIC,
                              "lanczos":Image.LANCZOS,
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
    
    def filter_exist(self, metadata, warning_threshold):
        """filter metadata based on existing images"""
        len_expected = len(metadata)
        existing_files = set(os.listdir(SAVE_DIR))
        indices_to_keep = []
        for i in range(len(metadata)):
            img_hash = metadata[i]['hash']
            filename = f"{img_hash}.jpg"
            if filename in existing_files:
                indices_to_keep.append(i)
        metadata = metadata.select(indices_to_keep)
        print(f"Filtered metadata, remaining items: {len(metadata)}")
        len_real = len(metadata)
        assert len_real >= len_expected * warning_threshold, f"Filtered metadata is too small: {len_real} < {len_expected} * {warning_threshold}. Either download images or adjust your expectation."
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_hash = item['hash']
        img_path = os.path.join(SAVE_DIR, f"{img_hash}.jpg")

        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0) # sd set image range to [-1, 1]

        item_out = dict(item)
        item_out['jpg'] = image
        item_out['txt'] = item_out['TEXT']
        return item_out
    

def caption_func(caption_type, image, size, img_hash):
    caption_image_path = os.path.join(CAPTION_DIR, caption_type, str(size), f"{img_hash}.jpg")
    if os.path.exists(caption_image_path):
        caption_image = Image.open(caption_image_path)
        if not caption_image.mode == "RGB":
            caption_image = caption_image.convert("RGB")
        caption_image = np.array(caption_image).astype(np.float32)
        caption_image = (caption_image / 127.5 - 1.0) # sd set image range to [-1, 1]
        return caption_image
    else:
        function_map = {
            'canny': caption_canny,
            'mlsd': caption_mlsd,
            'scribble': caption_scribble,
            'softedge': caption_softedge,
            'depth': caption_depth,
            'normal': caption_normal,
            'seg': caption_seg,
            'openpose': caption_openpose,
            'linear': caption_linear,
            'lineartanime': caption_lineartanime,
            'shuffle': caption_shuffle,
        }
        func = function_map[caption_type]
        caption_image = func(image, size) # (H, W, 3), np.uint8
        os.makedirs(os.path.dirname(caption_image_path), exist_ok=True)
        caption_image_pil = Image.fromarray(caption_image)
        caption_image_pil.save(caption_image_path)
        caption_image = (caption_image.astype(np.float32) / 127.5 - 1.0) # sd set image range to [-1, 1]
        return caption_image


class LaionArt_Caption(LaionArt_Base):
    def __init__(self,caption=['canny'], pre_caption = True, lang = ['en'], top = 0, bottom = 1_000_000, split = 'train', split_rate=[0.9,0.05,0.05], warning_threshold=0.8, size = 512, interpolation="bicubic"):
        assert isinstance(caption, list), "caption must be a list"
        assert all([c in ['canny', 'mlsd', 'scribble', 'softedge', 'depth', 'normal', 'seg', 'openpose', 'linear', 'lineartanime', 'shuffle', 'inpaint'] for c in caption]), "caption must be one of ['canny', 'mlsd', 'scribble', 'softedge', 'depth', 'normal', 'seg', 'openpose', 'linear', 'lineartanime', 'shuffle', 'inpaint']"
        self.caption = caption
        super().__init__(lang, top, bottom, split, split_rate, warning_threshold, size, interpolation)
        if pre_caption:
            print("Pre-generating captions...")
            self.initialize_captions()
            print("Pre-generation done.")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        img_hash = item['hash']
        image = ((item['jpg'] + 1.0) * 127.5).astype(np.uint8) # convert back to [0, 255] for captioning
        for c in self.caption:
            caption_image = caption_func(c, image, self.size, img_hash)
            item[c] = caption_image
        return item
    
    def initialize_captions(self):
        for i in tqdm(range(len(self))):
            _ = self.__getitem__(i)
        clean_models()


if __name__=='__main__':
    a = LaionArt_Caption(caption=['canny'], bottom= 1_000, split='train')
    b = a.__getitem__(0)
    print(b.keys())
