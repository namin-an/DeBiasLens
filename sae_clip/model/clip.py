import os
import urllib
from tqdm import tqdm
from typing import Union, List
import torch
from torch import nn
from sae_clip.model.model import SAECLIP
import clip
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



_MODELS = {
    "ViT-B/16-gender": {
     "url": "http://www.robots.ox.ac.uk/~maxbain/oxai-bias/best_ndkl_oai-clip-vit-b-16_neptune_run_OXVLB-317_model_e4_step_5334_embeddings.pt",
     "clip_arch": "ViT-B/16",
     "num_debias_tokens": 2
    },
    "ViT-L/14@336px": {
        "url": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        "clip_arch": "ViT-L/14@336px",
        "num_debias_tokens": 2
    },
    "ViT-B/16": {
        "url": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "clip_arch": "ViT-B/16",
        "num_debias_tokens": 2
    },
    "clip-vit-base-patch16":{
        "num_debias_tokens": 2
    },
    "clip-vit-large-patch14-336":{
        "num_debias_tokens": 2
    }
}

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    print(f"Installing pretrained embedings\n {url.split('/')[-1]}...")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None,
         layer: int = 23, layer_txt: int = 11, is_vision: bool = False):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name == 'InternViT-300M-448px':
        import sys
        sys.path.append('/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/models')
        from internlm import InternViT
        internlm = InternViT(name, device, layer, layer_txt)
        return internlm, internlm.processor
    
    elif name == 'llava-onevision-qwen2-7b-ov-hf':
        import sys
        sys.path.append('/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/models')
        from llavaonevision import LlavaOneVision
        model = LlavaOneVision(name, device, layer, layer_txt, is_vision)
        return model, model.processor

    else:
        cache_dir = download_root or os.path.expanduser("~/.cache/sae_clip")

        if name in _MODELS:
            model_info = _MODELS[name]
            filename = os.path.basename(model_info["url"])
            model_path = os.path.join(cache_dir, filename)

            # ✅ Only download if not already cached
            if not os.path.exists(model_path):
                print(f"Downloading model to {model_path}...")
                model_path = _download(model_info["url"], cache_dir)
            else:
                print(f"Using cached model at {model_path}")
                
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

        with open(model_path, 'rb') as opened_file:
            pt_embeddings = torch.load(opened_file, map_location="cpu", weights_only=False)

        # if is_text:
        #     # import sys
        #     # sys.path.append('/workspace/cvml_user/namin/bias_vlm/sae-for-vlm')
        #     # from models.clip import Clip
        #     # model = Clip(name, device, True)
        #     # preprocess = model.processor
        # else:
        clip_model, preprocess = clip.load(_MODELS[name]["clip_arch"], device=device)
        hidden_dim = clip_model.token_embedding.weight.shape[1]
        
        model = SAECLIP(clip_model=clip_model, num_debias_tokens=_MODELS[name]["num_debias_tokens"], hidden_dim=hidden_dim, layer=layer, layer_txt=layer_txt)
        try:
            model.debias_tokens.weight = nn.Parameter(pt_embeddings) # load pt embeddings
        except:
            print('No debiasing tokens')

        return model, preprocess