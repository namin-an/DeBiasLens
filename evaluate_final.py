import os
import argparse
import random
from tqdm import tqdm
import json
import ast
import pathlib
import pandas as pd
from collections import defaultdict
import clip
import warnings
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
from transformers import BlipProcessor, AutoProcessor, BlipForImageTextRetrieval

import sae_clip

import sys
sys.path.append('../sae-for-vlm')
from models.llava import Llava
from dictionary_learning.trainers import MatroyshkaBatchTopKSAE


def main(args):
    if args.stereotype:
        prompts = ["stereotype"]
    else:
        prompts = ["adjectives"] #, "occupations", "activities"]  
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
        deb_clip_model, preprocess = sae_clip.load("ViT-B/16-gender", device=device, layer=11)
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
        epoch = 100000
        expansion_factor = 8 
        layer = 11
        
        if args.text:
            eval_data_type = 'fairface'
            data_type = 'cocogendertxt'
            clip_type = 'vit-base-patch16'   #'vit-large-patch14-336' #'vit-base-patch16' 
            md_flag = "ViT-B/16" #"ViT-L/14@336px" #"ViT-B/16"
            is_image = False
            neurons_to_fix1 = {} #{1200:0, 1057:0} #{}
        else:
            eval_data_type = args.data_type # 'fairface'
            data_type = 'fairface'
            clip_type = 'vit-base-patch16' #'vit-large-patch14-336' #'vit-base-patch16'
            md_flag = "ViT-B/16" #"ViT-L/14@336px" #"ViT-B/16"
            is_image = True
            # neurons_to_fix1 = {98:0, 195:0}#{98:0, 195:0} #{98:0, 195:0} #{458:0, 375:0} #{98:0, 195:0}
            # neurons_to_fix2 = {105:0, 163:0, 102:0}
            # neurons_to_fix3 = {6:0, 264:0, 279:0, 98:0, 454:0, 364:0}
            # neurons_to_fix1 = {465:0, 91:0, 832:0, 402:0, 131:0, 167:0, 226:0}
            # layer = 23
    
            
        # llava = Llava() 
        sae_path = f"[your_working_path]/DeBiasLens/sae-for-vlm/checkpoints_dir/matroyshka_batch_top_k_20_x{expansion_factor}/random_k_2/{data_type}_train_activations_clip-{clip_type}_{layer}_post_mlp_residual_matroyshka_batch_top_k_20_x{expansion_factor}/trainer_0/checkpoints/ae_{epoch}.pt"
        sae = MatroyshkaBatchTopKSAE.from_pretrained(sae_path).cuda()
        
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

    if args.data_type in ['fairface', 'pata', 'cocogender']:
        attribute_lst = ['gender', 'age', 'race'] 
    elif args.data_type == 'pairs':
        attribute_lst = ['gender', 'race']

    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if sae == None:
            for attribute in attribute_lst:
                for prmpt in prompts:
                    print(prmpt)
                    val, df = sae_clip.measure_bias(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, attribute=attribute,
                                                    prompt=prmpt, mod=args.mod, md_flag=args.md_flag, data_type=args.data_type,
                                                    is_siglip=is_siglip, is_blip=is_blip,
                                                    suppress_list=suppress_list, excite_dict=excite_dict,
                                                    sae=sae)
                    print(val)
                    df.to_csv(f'summaries_{args.data_type}_{args.md_flag}_{attribute}_{prmpt}.csv')
        
        else:
            # neurons_to_fix1 = {4290:0, 3374:0, 648:0, 5052:0, 2277:0, 2887:0, 3896:0, 5458:0, 1321:0, 2327:0, 2675:0, 416:0, 5171:0, 1810:0, 4594:0, 1128:0, 1285:0, 4297:0} # random
            neurons_to_fix1 = {105:0, 155:0, 340:0, 196:0, 335:0, 317:0, 749:0, 87:0, 266:0, 82:0, 85:0, 285:0, 696:0, 17:0, 163:0, 927:0, 337:0, 280:0, 84:0, 102:0, 36:0, 306:0, 276:0, 418:0, 124:0} # age
            # neurons_to_fix1 = {105:0, 340:0, 196:0, 317:0, 749:0, 87:0, 82:0, 696:0, 17:0, 163:0, 337:0, 84:0, 102:0, 306:0, 124:0} # age_filtered
            neurons_to_fix2 = {6:0, 831:0, 264:0, 135:0, 279:0, 358:0, 98:0, 155:0, 192:0, 328:0, 454:0, 364:0, 522:0} # race
            neurons_to_fix3 = {98:0, 11:0, 124:0, 354:0, 203:0, 195:0, 236:0, 242:0, 254:0, 93:0, 733:0, 46:0, 302:0, 114:0, 117:0, 280:0, 331:0} # gender
            
            neurons_to_fix1 = neurons_to_fix1 | neurons_to_fix2 | neurons_to_fix3

            deb_clip_model, preprocess = sae_clip.load(md_flag, device=device, layer=layer)            
            try:
                deb_clip_model.eval()
            except:
                deb_clip_model.model.eval()
            for attribute in attribute_lst:
                for prmpt in prompts:
                    for alpha in [0.6, 1.0]: #[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                        deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix1, pre_zero=False, alpha=alpha, is_image=is_image)
                        val, df = sae_clip.measure_bias(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, attribute=attribute,
                                                    prompt=prmpt, mod=args.mod, md_flag=args.md_flag, data_type=eval_data_type,
                                                    is_siglip=is_siglip, is_blip=is_blip, epoch=epoch,
                                                    suppress_list=suppress_list, excite_dict=excite_dict,
                                                    sae=sae)
                        print(alpha, val)
                        df.to_csv(f'summaries_{eval_data_type}_{args.md_flag}_{attribute}_{prmpt}.csv')
        
            
            # if not args.text:
            #     deb_clip_model, preprocess = sae_clip.load(md_flag, device=device, layer=layer)
            #     try:
            #         deb_clip_model.eval()
            #     except:
            #         deb_clip_model.model.eval()
            #     for attribute in attribute_lst:
            #         for prmpt in prompts:
            #             for alpha in [0.6]: #, 1.0]: #[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            #                 deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix2, pre_zero=False, alpha=alpha, is_image=is_image)
            #                 val, df = sae_clip.measure_bias(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, attribute=attribute,
            #                                             prompt=prmpt, mod=args.mod, md_flag=args.md_flag, data_type=eval_data_type,
            #                                             is_siglip=is_siglip, is_blip=is_blip, epoch=epoch,
            #                                             suppress_list=suppress_list, excite_dict=excite_dict,
            #                                             sae=sae)
            #                 print(alpha, val)
                
                # deb_clip_model, preprocess = sae_clip.load(md_flag, device=device, layer=layer)
                # try:
                #     deb_clip_model.eval()
                # except:
                #     deb_clip_model.model.eval()
                # for attribute in attribute_lst:
                #     for prmpt in prompts:
                #         for alpha in [0.6]: #, 1.0]: #[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                #             deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix3, pre_zero=False, alpha=alpha, is_image=is_image)
                #             val, df = sae_clip.measure_bias(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, attribute=attribute,
                #                                         prompt=prmpt, mod=args.mod, md_flag=args.md_flag, data_type=eval_data_type,
                #                                         is_siglip=is_siglip, is_blip=is_blip, epoch=epoch,
                #                                         suppress_list=suppress_list, excite_dict=excite_dict,
                #                                         sae=sae)
                #             print(alpha, val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mod", type=str, default=None, help="Options: None, vision, lang, projection")
    parser.add_argument("--md_flag", type=str, default='clip', help="Options: clip, blip, siglip, debias_clip, sae_clip")
    parser.add_argument("--category", type=str, default='gender', help="Options: gender, skin")
    parser.add_argument("--data_type", type=str, default='fairface', help="Options: fairface, pairs, pata")
    parser.add_argument("--excite_strength", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--neuron", action='store_true')
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--ckpt_dir", type=str, default="[your_working_path]/DeBiasLens/checkpoints")
    parser.add_argument("--text", action='store_true')
    parser.add_argument("--stereotype", action='store_true')

    args = parser.parse_args()
    main(args)