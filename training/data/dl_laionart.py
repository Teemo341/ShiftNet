import os
from datasets import load_dataset, load_from_disk, Dataset
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from PIL import Image, ImageStat

parser = argparse.ArgumentParser(description="Download images from LAION-Art dataset.")
parser.add_argument("--function", type=str, nargs='+', default=["download", "filter"], help="Function to execute.")
parser.add_argument("--data_dir", type=str, default="data/laionart", help="Directory to store the dataset.")
parser.add_argument("--num_threads", type=int, default=64, help="Number of threads for downloading images.")
parser.add_argument("--start_idx", type=int, default=0, help="Start index for the subset.")
parser.add_argument("--end_idx", type=int, default=3274199, help="End index for the subset.")
parser.add_argument("--languages", type=str, nargs='+', default=["en"], help="Languages to filter the dataset.")
args = parser.parse_args()

# 配置参数
DATA_DIR= args.data_dir
assert os.path.exists(DATA_DIR), f"Data directory {DATA_DIR} does not exist."
CACHE_DIR = f"{DATA_DIR}/laion-art-cache"
SAVE_DIR = f"{DATA_DIR}/images"
FILTER_DIR = f"{DATA_DIR}/filtered"
LOG_FAILED = f"{DATA_DIR}/failed_downloads.txt"
NUM_THREADS = args.num_threads

START_IDX = args.start_idx
END_IDX = args.end_idx
LANGUAGES = args.languages

SUBSET_PATH = f"{DATA_DIR}/subset_metadata/top_{LANGUAGES}_{START_IDX}_{END_IDX}"

# 检查图片有效性的函数
def check_image_validity(image_path, min_size=(16, 16), min_colors=10, min_filesize=1024):
    """
    检查图片有效性
    - 是否损坏
    - 是否极小尺寸
    - 是否纯色/伪装图（如黑底白字）
    - 文件大小是否太小
    """

    # 1. 文件大小检查
    if not os.path.exists(image_path) or os.path.getsize(image_path) < min_filesize:
        return False, "文件过小或不存在"
    try:
        # 2. 能否打开
        img = Image.open(image_path)
        # 3. 能否验证
        img.verify()
    except Exception as e:
        return False, f"无法打开或已损坏: {str(e)}"
    # 4. 再次打开以便分析像素（verify后img不能再用）
    try:
        img = Image.open(image_path)
    except Exception as e:
        return False, f"再次打开失败: {str(e)}"
    # 5. 检查尺寸
    if img.width < min_size[0] or img.height < min_size[1]:
        return False, f"图片尺寸过小: {img.width}x{img.height}"
    # 6. 统计颜色数（检测纯色或极少颜色）
    try:
        colors = img.convert("RGB").getcolors(maxcolors=256*256)
        if colors is not None and len(colors) < min_colors:
            return False, f"颜色种类过少: {len(colors)}"
    except Exception as e:
        pass  # 有些大图 getcolors 可能报错

    return True, "图片正常"

if __name__ == "__main__":

    if "download" in args.function:
        # 步骤1：加载或生成 subset
        if os.path.exists(SUBSET_PATH):
            print(f"Loading subset from disk: {SUBSET_PATH}")
            subset = load_from_disk(SUBSET_PATH)
            print(f"Loaded {len(subset)} items.")
        else:
            print("Loading laion-art dataset and generating subset...")
            ds = load_dataset("laion/laion-art", cache_dir=CACHE_DIR)
            print(f"Dataset loaded with {len(ds['train'])} items.")
            print(f"Filtering languages: {LANGUAGES}")
            filtered = ds['train'].filter(lambda x: x['LANGUAGE'] in LANGUAGES)
            print(f"Filtered dataset contains {len(filtered)} items.")
            if len(filtered) < END_IDX:
                raise ValueError(f"Filtered dataset has only {len(filtered)} items, which is less than END_IDX ({END_IDX}). Please adjust START_IDX and END_IDX.")
            print(f"Selecting from {START_IDX} to {END_IDX} (total {END_IDX-START_IDX})...")
            subset = filtered.sort('aesthetic', reverse=True).select(range(START_IDX, END_IDX))
            print(f"Selected {len(subset)} items. Saving subset to disk...")
            subset.save_to_disk(SUBSET_PATH)
            del filtered
            del ds

        os.makedirs(SAVE_DIR, exist_ok=True)

        # 步骤2：构建下载任务列表（断点续传）
        print("构建下载任务列表...")
        existing_files = set(os.listdir(SAVE_DIR))
        print(f"已存在图片数量: {len(existing_files)}")
        tasks = []
        for item in tqdm(subset, desc="Processing items", total=len(subset), leave=False):
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

if "filter" in args.function:
        FILTER_FILE = f"{FILTER_DIR}/ok_list.json"
        FILTER_CACHE = f"{FILTER_DIR}/filtered_cache"
        BAD_FILE = f"{FILTER_DIR}/bad_list.json"
        os.makedirs(FILTER_DIR, exist_ok=True)
        os.makedirs(FILTER_CACHE, exist_ok=True)

        # 加载已知正常图片列表
        if not os.path.exists(SAVE_DIR) or not os.listdir(SAVE_DIR):
            raise ValueError(f"Image directory {SAVE_DIR} is empty. Please download images first.")

        existing_files = set(os.listdir(SAVE_DIR))
        if os.path.exists(FILTER_FILE):
            print('loading from last time filtering')
            with open(FILTER_FILE, "r", encoding="utf-8") as f:
                ok_list = json.load(f)
        else:
            ok_list = []

        bad_dict = {}

        print(f"已下载图片数量: {len(existing_files)}, 已知正常图片数量: {len(ok_list)}")

        # 除去正常图片和已知坏图
        existing_files = existing_files - set(ok_list)
        print(f"待检查图片数量: {len(existing_files)}")
        if not existing_files:
            print("所有图片都已通过检查。")
        else:
            def check_one(filename):
                img_path = os.path.join(SAVE_DIR, filename)
                is_valid, reason = check_image_validity(img_path)
                return filename, is_valid, reason

            results = []
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = {executor.submit(check_one, filename): filename for filename in existing_files}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Checking images", leave=False):
                    filename, is_valid, reason = future.result()
                    results.append((filename, is_valid, reason))

            # 主线程收集和处理结果
            failed_remove = []
            for filename, is_valid, reason in results:
                img_path = os.path.join(SAVE_DIR, filename)
                if is_valid:
                    ok_list.append(filename)
                else:
                    print(f"图片 {filename} 无效: {reason}")
                    bad_dict[filename] = reason
                    # 将无效图片移动到过滤目录，并删除原图
                    invalid_path = os.path.join(FILTER_CACHE, filename)
                    try:
                        os.rename(img_path, invalid_path)
                    except Exception as e:
                        print(f"移动文件出错: {filename} -> {invalid_path}, 错误: {e}")
                        failed_remove.append(f"移动文件出错: {filename} -> {invalid_path}, 错误: {e}")

            # 保存正常图片列表
            with open(FILTER_FILE, "w", encoding="utf-8") as f:
                json.dump(ok_list, f, ensure_ascii=False, indent=4)
            print(f"已保存正常图片列表到 {FILTER_FILE}，共 {len(ok_list)} 张图片。")
            # 保存无效图片说明
            with open(BAD_FILE, "w", encoding="utf-8") as f:
                json.dump(bad_dict, f, ensure_ascii=False, indent=4)
            print(f"已保存无效图片列表到 {BAD_FILE}，共 {len(bad_dict)} 张图片。")
            if failed_remove:
                print("以下文件移动失败：")
                for msg in failed_remove:
                    print(msg)
            else:
                print("所有无效图片已成功移动到过滤目录。")
            
