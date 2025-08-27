# Uniformer
# From https://github.com/Sense-X/UniFormer
# # Apache-2.0 license

import os

from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette
from annotator.util import annotator_ckpts_path


checkpoint_file = "https://huggingface.co/lllyasviel/Annotators/resolve/main/upernet_global_small.pth"


class UniformerDetector:
    def __init__(self):
        modelpath = os.path.join(annotator_ckpts_path, "upernet_global_small.pth")
        if not os.path.exists(modelpath):
            import wget
            os.makedirs(annotator_ckpts_path, exist_ok=True)
            print(f'Downloading {checkpoint_file} to {modelpath}...')
            wget.download(checkpoint_file, modelpath)
            print('Download complete.')
        config_file = os.path.join(os.path.dirname(annotator_ckpts_path), "uniformer", "exp", "upernet_global_small", "config.py")
        self.model = init_segmentor(config_file, modelpath).cuda()

    def __call__(self, img):
        result = inference_segmentor(self.model, img)
        res_img = show_result_pyplot(self.model, img, result, get_palette('ade'), opacity=1)
        return res_img
