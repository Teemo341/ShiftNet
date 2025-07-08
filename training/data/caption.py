"""pipline for captioning images using various methods"""
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random

from annotator.util import nms, resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.hed import HEDdetector
from annotator.pidinet import PidiNetDetector
from annotator.midas import MidasDetector
from annotator.zoe import ZoeDetector
from annotator.normalbae import NormalBaeDetector
from annotator.uniformer import UniformerDetector
from annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector
from annotator.openpose import OpenposeDetector
from annotator.lineart import LineartDetector
from annotator.lineart_anime import LineartAnimeDetector
from annotator.shuffle import ContentShuffleDetector


# global variables
seed = 3407

canny_processor = None
mlsd_processor = None
scribble_processor = None
softedge_processor = None
depth_processor = None
normal_processor = None
seg_processor = None
openpose_processor = None
lineart_processor = None
lineartanime_processor = None
shuffle_processor = None


#! edge based
def caption_canny(input_image, resolution = 512, canny_low_threshold=100, canny_high_threshold=200):
    global canny_processor
    if not isinstance(canny_processor, CannyDetector):
        canny_processor = CannyDetector()
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = canny_processor(resize_image(input_image, resolution), canny_low_threshold, canny_high_threshold)
        detected_map = HWC3(detected_map)
    return detected_map

def caption_mlsd(input_image,resolution = 512, value_threshold=0.1, distance_threshold=0.1):
    global mlsd_processor
    if not isinstance(mlsd_processor, MLSDdetector):
        mlsd_processor = MLSDdetector()
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = mlsd_processor(resize_image(input_image, resolution), value_threshold, distance_threshold)
        detected_map = HWC3(detected_map)
    return detected_map

def caption_scribble(input_image, resolution = 512, det='HED'):
    # det = 'HED' or 'PIDI' or 'None'
    global scribble_processor
    if 'HED' in det:
        if not isinstance(scribble_processor, HEDdetector):
            scribble_processor = HEDdetector()
    elif 'PIDI' in det:
        if not isinstance(scribble_processor, PidiNetDetector):
            scribble_processor = PidiNetDetector()
    else:
        raise ValueError(f"Unknown scribble detector: {det}")
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = scribble_processor(resize_image(input_image, resolution))
        detected_map = HWC3(detected_map)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0
    return detected_map

def caption_softedge(input_image, resolution = 512, det='SoftEdge_PIDI_safe'):
    # Robustness: SoftEdge_PIDI_safe > SoftEdge_HED_safe >> SoftEdge_PIDI > SoftEdge_HED
    # Maximum result quality: SoftEdge_HED > SoftEdge_PIDI > SoftEdge_HED_safe > SoftEdge_PIDI_safe
    global softedge_processor
    if 'HED' in det:
        if not isinstance(softedge_processor, HEDdetector):
            softedge_processor = HEDdetector()
    elif 'PIDI' in det:
        if not isinstance(softedge_processor, PidiNetDetector):
            softedge_processor = PidiNetDetector()
    else:
        raise ValueError(f"Unknown softedge detector: {det}")
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = softedge_processor(resize_image(input_image, resolution), safe='safe' in det)
        detected_map = HWC3(detected_map)
    return detected_map


#! segmentation based
def caption_depth(input_image, resolution = 512, det='Depth_Midas'):
    # det = 'Depth_Midas' or 'Depth_Zoe'
    global depth_processor
    if 'Midas' in det:
        if not isinstance(depth_processor, MidasDetector):
            depth_processor = MidasDetector()
    elif 'Zoe' in det:
        if not isinstance(depth_processor, ZoeDetector):
            depth_processor = ZoeDetector()
    else:
        raise ValueError(f"Unknown depth detector: {det}")
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = depth_processor(resize_image(input_image, resolution))
        detected_map = HWC3(detected_map)
    return detected_map

def caption_normal(input_image, resolution = 512):
    global normal_processor
    if not isinstance(normal_processor, NormalBaeDetector):
        normal_processor = NormalBaeDetector()
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = normal_processor(resize_image(input_image, resolution))
        detected_map = HWC3(detected_map)
    return detected_map

def caption_seg(input_image, resolution = 512, det='Seg_OFADE20K'):
    global seg_processor
    if det == 'Seg_OFCOCO':
        if not isinstance(seg_processor, OneformerCOCODetector):
            seg_processor = OneformerCOCODetector()
    elif det == 'Seg_OFADE20K':
        if not isinstance(seg_processor, OneformerADE20kDetector):
            seg_processor = OneformerADE20kDetector()
    elif det == 'Seg_UFADE20K':
        if not isinstance(seg_processor, UniformerDetector):
            seg_processor = UniformerDetector()
    else:
        raise ValueError(f"Unknown segmentation detector: {det}")
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = seg_processor(resize_image(input_image, resolution))
        detected_map = HWC3(detected_map)
    return detected_map


#! backbone based
def caption_openpose(input_image, resolution = 512, det='Openpose_full'):
    # det = 'Openpose' or 'Openpose_Full', where 'Full' includes hand and face detection
    global openpose_processor
    if not isinstance(openpose_processor, OpenposeDetector):
        openpose_processor = OpenposeDetector()
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = openpose_processor(resize_image(input_image, resolution), hand_and_face='Full' in det)
        detected_map = HWC3(detected_map)
    return detected_map


#! linear based
def caption_linear(input_image, resolution = 512, det = 'Lineart'):
    # det = 'Lineart' or 'Lineart_Coarse'
    global lineart_processor
    if not isinstance(lineart_processor, LineartDetector):
        lineart_processor = LineartDetector()
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = lineart_processor(resize_image(input_image, resolution), coarse='Coarse' in det)
        detected_map = HWC3(detected_map)
    return detected_map

def caption_lineartanime(input_image, resolution = 512):
    global lineartanime_processor
    if not isinstance(lineartanime_processor, LineartAnimeDetector):
        lineartanime_processor = LineartAnimeDetector()
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = lineartanime_processor(resize_image(input_image, resolution))
        detected_map = HWC3(detected_map)
    return detected_map


#! original image based
def caption_shuffle(input_image, resolution = 512):
    global shuffle_processor
    if not isinstance(shuffle_processor, ContentShuffleDetector):
        shuffle_processor = ContentShuffleDetector()
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = input_image.copy()
        img = resize_image(input_image, resolution)
        H, W, C = img.shape
        np.random.seed(seed)
        detected_map = shuffle_processor(detected_map, w=W, h=H, f=256)
    return detected_map

def random_mask(height, width):
    """生成随机自由形状遮挡mask(类似LAMA/random mask)"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(random.randint(5, 15)):
        shape_type = random.choice(['rect', 'ellipse', 'poly'])
        if shape_type == 'rect':
            x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
            x2, y2 = random.randint(x1+10, width), random.randint(y1+10, height)
            draw.rectangle([x1, y1, x2, y2], fill=255)
        elif shape_type == 'ellipse':
            x1, y1 = random.randint(0, width-20), random.randint(0, height-20)
            x2, y2 = x1+random.randint(10, 40), y1+random.randint(10, 40)
            draw.ellipse([x1, y1, x2, y2], fill=255)
        elif shape_type == 'poly':
            num_points = random.randint(3, 8)
            points = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_points)]
            draw.polygon(points, fill=255)
    return np.array(mask) // 255

def random_occlusion_mask(height, width):
    """生成光流遮挡风格的mask(带方向、条带、不规则块状遮挡)"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # 生成若干条带
    for _ in range(random.randint(1, 4)):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        angle = random.uniform(0, 2*np.pi)
        length = random.randint(width//2, width)
        thickness = random.randint(15, 50)
        # 计算带状的起止点
        x_end = int(x + length * np.cos(angle))
        y_end = int(y + length * np.sin(angle))
        draw.line([x, y, x_end, y_end], fill=255, width=thickness)
        # 可选：在线条周围画一些椭圆增加不规则性
        for _ in range(random.randint(3, 8)):
            offset_x = random.randint(-thickness, thickness)
            offset_y = random.randint(-thickness, thickness)
            xx = int(x + offset_x)
            yy = int(y + offset_y)
            rr = random.randint(thickness//2, thickness)
            draw.ellipse([xx-rr, yy-rr, xx+rr, yy+rr], fill=255)
    # 可能再加一些小块
    for _ in range(random.randint(1, 3)):
        x, y = random.randint(0, width-20), random.randint(0, height-20)
        w, h = random.randint(10, 30), random.randint(10, 30)
        draw.rectangle([x, y, x+w, y+h], fill=255)
    return np.array(mask) // 255

def caption_inpaint(input_image, resolution = 512): #! only return mask, 1 for inpaint area, 0 for valid area
    input_image = HWC3(input_image)
    H, W, C = input_image.shape
    if random.random() < 0.5:
        mask=random_mask(H, W)
    else:
        mask=random_occlusion_mask(H, W)
    return mask


if __name__ == "__main__":
    # Example usage
    input_image = cv2.imread('./test_imgs/bird.png')
    canny_map = caption_canny(input_image)
    mlsd_map = caption_mlsd(input_image)
    scribble_map = caption_scribble(input_image, det='HED')
    softedge_map = caption_softedge(input_image, det='SoftEdge_PIDI_safe')
    depth_map = caption_depth(input_image, det='Depth_Midas')
    normal_map = caption_normal(input_image)
    seg_map = caption_seg(input_image, det='Seg_OFADE20K')
    openpose_map = caption_openpose(input_image, det='Openpose_full')
    lineart_map = caption_linear(input_image, det='Lineart_Coarse')
    lineartanime_map = caption_lineartanime(input_image)
    shuffle_map = caption_shuffle(input_image)
    
    # Display results
    cv2.imshow('Canny', canny_map)
    cv2.imshow('MLSD', mlsd_map)
    cv2.imshow('Scribble', scribble_map)
    cv2.imshow('SoftEdge', softedge_map)
    cv2.imshow('Depth', depth_map)
    cv2.imshow('Normal', normal_map)
    cv2.imshow('Segmentation', seg_map)
    cv2.imshow('Openpose', openpose_map)
    cv2.imshow('Lineart', lineart_map)
    cv2.imshow('Lineart Anime', lineartanime_map)
    cv2.imshow('Shuffle', shuffle_map)
    
    cv2.waitKey(0)