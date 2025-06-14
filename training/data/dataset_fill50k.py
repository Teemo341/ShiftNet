import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset

DATA_DIR= "data/fill50k"
assert os.path.exists(DATA_DIR), f"Data directory {DATA_DIR} does not exist."

class fill50k(Dataset):
    def __init__(self):
        self.data = []
        with open(f'{DATA_DIR}/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(f'{DATA_DIR}/' + source_filename)
        target = cv2.imread(f'{DATA_DIR}/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

if __name__ == '__main__':
    dataset = fill50k()
    print(f"Dataset length: {len(dataset)}")

    item = dataset[1234]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(f"Text: {txt}")
    print(f"Image shape: {jpg.shape}")
    print(f"Hint shape: {hint.shape}")