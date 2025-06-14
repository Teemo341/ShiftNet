import os
from datasets import load_dataset, load_from_disk, Dataset
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置参数
DATA_DIR= "data/laionart"
assert os.path.exists(DATA_DIR), f"Data directory {DATA_DIR} does not exist."
CACHE_DIR = f"{DATA_DIR}/laion-art-cache"
SAVE_DIR = f"{DATA_DIR}/images"
LOG_FAILED = f"{DATA_DIR}/failed_downloads.txt"
NUM_THREADS = 64

START_IDX = 0
END_IDX = 1_000_000
LANGUAGES = ["en"]

SUBSET_PATH = f"{DATA_DIR}/subset_metadata/top_{LANGUAGES}_{START_IDX}_{END_IDX}"
# 步骤1：加载或生成 subset
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
    del filtered
    del ds

os.makedirs(SAVE_DIR, exist_ok=True)

# 步骤2：构建下载任务列表（断点续传）
existing_files = set(os.listdir(SAVE_DIR))
tasks = []
for item in subset:
    img_hash = item['hash']
    url = item['URL']
    filename = f"{img_hash}.jpg"
    if filename not in existing_files:
        tasks.append((url, filename))

print(f"待下载图片数量: {len(tasks)}")

# 步骤3：定义下载函数
def download_image(url, filename):
    img_path = os.path.join(SAVE_DIR, filename)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; LAION-Downloader/1.0)'}
        resp = requests.get(url, timeout=10, headers=headers)
        if resp.status_code == 200 and resp.content:
            if resp.headers.get('Content-Type', '').startswith('image'):
                with open(img_path, "wb") as f:
                    f.write(resp.content)
                return True
        return False
    except Exception as e:
        # print(f"Exception downloading {url}: {e}")
        return False

# 步骤4：多线程并发下载
try:
    with open(LOG_FAILED, "a") as logf:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            future_to_task = {executor.submit(download_image, url, filename): (url, filename) for url, filename in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Downloading images", leave=False):
                url, filename = future_to_task[future]
                try:
                    success = future.result()
                    if not success:
                        logf.write(f"{filename}\t{url}\n")
                        logf.flush()
                except Exception:
                    logf.write(f"{filename}\t{url}\n")
                    logf.flush()
    print("下载任务完成。")
except KeyboardInterrupt:
    print("手动终止下载，已保存进度。")