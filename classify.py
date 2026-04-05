import argparse
import os
import ast
import pathlib
import pandas as pd
import json
from collections import defaultdict
import clip
from PIL import Image
from tqdm import tqdm
import torch
import warnings
from transformers import AutoProcessor, AutoModel
from transformers import BlipForQuestionAnswering, BlipProcessor, AutoProcessor, BlipForImageTextRetrieval

import sae_clip
from sae_clip.model.model import ClipLike, model_loader

import sys
sys.path.append('[your_working_path]/DeBiasLens/sae-for-vlm')
from dictionary_learning.trainers import MatroyshkaBatchTopKSAE


def main(args):
    if args.neuron:
        if args.category == 'gender':
            concepts = ['Male', 'Female']
        elif args.category == 'skin':
            concepts = ['Light', 'Dark']

        SRC_PATH = (pathlib.Path(__file__) / ".." / "..").resolve().absolute()

        DATA_PATH = SRC_PATH / f'bias_vlm/scripts/neurons_cocogender_{args.md_flag}_{args.mod}_{args.category}_100.csv'
        df = pd.read_csv(DATA_PATH) 
        df.index = df['Unnamed: 0'].values
        df = df[concepts]
        
        cond1 = (df[concepts[0]] == 1) & (df[concepts[1]] == 0)
        cond2 = (df[concepts[0]] == 0) & (df[concepts[1]] == 1)
        
        indices1 = df[cond1].index # Male
        indices2 = df[cond2].index # Female
        
        print(f"# {concepts[0]} neurons: {len(indices1)}", f"# {concepts[1]} neurons: {len(indices2)}")
        
        suppress_list = [ast.literal_eval(x) for x in indices1] #indices1
        
        excite_dict = defaultdict(dict)
        for layer_neuron in indices2:
            layer_neuron = ast.literal_eval(layer_neuron)
            layer, neuron = int(layer_neuron[0]), int(layer_neuron[1])
            excite_dict[layer][neuron] = args.excite_strength
    else:
        suppress_list = None
        excite_dict = None
        
    
    device='cuda'
    if args.md_flag == 'clip':
        deb_clip_model, preprocess = sae_clip.load("ViT-L/14@336px", device=device)
    elif args.md_flag == 'debias_clip':
        deb_clip_model, preprocess = sae_clip.load("ViT-B/16-gender", device=device)
    elif args.md_flag == 'clip-vitb32':
        deb_clip_model, preprocess = clip.load("ViT-B/32", device=device)
        # deb_clip_model, preprocess, tokenizer, alias_name = model_loader(
        #     "openai/CLIP/ViT-B/32", device
        # )
    elif args.md_flag == 'clip-vitb16':
        deb_clip_model, preprocess = clip.load("ViT-B/16", device=device)
    elif args.md_flag == 'clip-vitl14':
        deb_clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)
    elif args.md_flag == 'siglip':
        deb_clip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        preprocess = AutoProcessor.from_pretrained("google/siglip-base-patch16-224") 
        deb_clip_model.to(device)
    elif args.md_flag == 'blip-itm':
        # deb_clip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        # preprocess = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        deb_clip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        preprocess = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        deb_clip_model.to(device)
    elif args.md_flag == 'sae_clip':
        data_type = "fairface" #"cocogender"
        epoch = 100000
        expansion_factor = 8 # 64 # 2
        layer = 11

        if args.text:
            data_type = 'cocogendertxt'
            clip_type = 'vit-base-patch16' #'vit-large-patch14-336'
            md_flag = "ViT-B/16" #"ViT-L/14@336px"
            is_image = False
            is_text = True
            neurons_to_fix1 = {}
        else:
            data_type = 'fairface'
            clip_type = 'vit-base-patch16'
            md_flag = "ViT-B/16"
            is_image = True
            is_text = False
            neurons_to_fix1 = {98:0, 195:0}
            neurons_to_fix2 = {105:0, 163:0, 102:0}
            neurons_to_fix3 = {6:0, 264:0, 279:0, 98:0, 454:0, 364:0}
        
        sae_path = f"[your_working_path]/DeBiasLens/sae-for-vlm/checkpoints_dir/matroyshka_batch_top_k_20_x{expansion_factor}/random_k_2/{data_type}_train_activations_clip-{clip_type}_{layer}_post_mlp_residual_matroyshka_batch_top_k_20_x{expansion_factor}/trainer_0/checkpoints/ae_{epoch}.pt"
        sae = MatroyshkaBatchTopKSAE.from_pretrained(sae_path).cuda()

        # neuron = 275
        # alpha = 10 #30 #1.5
        # deb_clip_model, preprocess = sae_clip.load("ViT-B/16", device=device, layer=layer) #"ViT-L/14@336px", device=device)
        # deb_clip_model.num_prompts_tokz = 0

        # ig_save_path = os.path.join(args.ckpt_dir, f"integrated_gradients_{data_type}.pt") #_epoch_{epoch+1}.pt")
        # igs = torch.load(ig_save_path) # igs shape: (num_classes, num_data, num_neurons)
        # top_k = 10  # for example, top 10 neurons per class
        # igs_mean = igs.mean(dim=1)
        # _, topk_indices = torch.topk(igs_mean, k=top_k, dim=1)

        #####
        # start_idx = 0
        # end_idx = 2048 #65536

        # save_dir = f"[your_working_path]/DeBiasLens/neuron_outputs_{data_type}"
        # file_prefix = f"neurons_{start_idx}_{end_idx}_{epoch}_{expansion_factor}"
        
        # # laion
        # file_name = "laion_400_unigram.txt" #"clip_disect_20k.txt"
        # file_path = f"[your_working_path]/DeBiasLens/MSAE/vocab/{file_name}"  # Replace with your file path
        # with open(file_path, "r", encoding="utf-8") as f:
        #     words = [line.strip() for line in f if line.strip()]
        
        # from LVLM
        # path = f"[your_working_path]/DeBiasLens/neuron_outputs_{data_type}/neurons_0_{end_idx}_{epoch}_{expansion_factor}.json"
        # words = []
        # with open(path, "r") as f:
        #     for line in f:
        #         if not line.strip():
        #             continue
        #         data = json.loads(line)
        #         response = data["response"].lower()
        #         words.append(response)
        # print(len(words))

        # # save vocabs
        # save_emb_path = os.path.join(save_dir, f"vocab_{file_prefix}.pt")
        # if not os.path.exists(save_emb_path):
        #     all_embeddings = []
        #     device='cuda'
        #     for vocab in tqdm(words):
        #         with torch.no_grad():
        #             txt = clip.tokenize(vocab).to(device)
        #             emb = deb_clip_model.encode_text(txt)
        #             emb /= emb.norm(dim=-1, keepdim=True)
        #         all_embeddings.append(emb.detach().cpu())
        #     vocab_embeddings = torch.stack(all_embeddings, dim=0)
        #     torch.save(vocab_embeddings, save_emb_path)
        # else:
        #     vocab_embeddings = torch.load(save_emb_path)

        # # save embeddings
        # save_emb_path = os.path.join(save_dir, f"embs_{file_prefix}.pt")
        # if not os.path.exists(save_emb_path):
        #     neurons_lst = list(range(start_idx, end_idx))
        #     all_embeddings = []
        #     for neuron in tqdm(neurons_lst):
        #         image = Image.new("RGB", (224, 224), color="white") 
        #         image = preprocess(image).unsqueeze(0).to(device)
        #         with torch.no_grad():
        #             deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix={neuron: alpha}, pre_zero=False)
        #             emb = deb_clip_model.encode_image(image)
        #             emb /= emb.norm(dim=-1, keepdim=True)
        #         all_embeddings.append(emb.detach().cpu())
        #     embs = torch.stack(all_embeddings, dim=0)
        #     torch.save(embs, save_emb_path)
        # else:
        #     embs = torch.load(save_emb_path)

        #####

        # deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix={neuron: alpha}, pre_zero=False)
    else:
        raise NotImplementedError

    # deb_clip_model.eval()
    if 'sae' not in args.md_flag:
        sae = None
    
    if args.md_flag == 'siglip':
        is_siglip = True
        is_blip = False
        tokenizer = preprocess
    elif 'blip' in args.md_flag:
        is_blip = True
        is_siglip = False
        tokenizer = preprocess
    else:
        is_siglip = False
        is_blip = False
        tokenizer = clip
    
    # measure bias, lower == less biased
    deb_clip_model, preprocess = sae_clip.load(md_flag, device=device, layer=layer)
    try:
        deb_clip_model.eval()
    except:
        deb_clip_model.model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix1, pre_zero=False, alpha=alpha, is_image=is_image)
            print(sae_clip.classify(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, 
                                    mod=args.mod, md_flag=args.md_flag, data_type=args.data_type,
                                    is_siglip=is_siglip, is_blip=is_blip,
                                    suppress_list=suppress_list, excite_dict=excite_dict,
                                    sae=sae))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mod", type=str, default=None, help="Options: None, vision, lang, projection")
    parser.add_argument("--md_flag", type=str, default='clip', help="Options: clip, blip, siglip, debias_clip, sae_clip")
    parser.add_argument("--category", type=str, default='gender', help="Options: gender, skin")
    parser.add_argument("--data_type", type=str, default='imagenet', help="Options: imagenet")
    parser.add_argument("--excite_strength", type=int, default=2)
    parser.add_argument("--neuron",  action='store_true')
    parser.add_argument("--sae",  action='store_true')
    parser.add_argument("--text", action='store_true')

    parser.add_argument("--ckpt_dir", type=str, default="[your_working_path]/DeBiasLens/checkpoints")
    
    args = parser.parse_args()
    main(args)