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
            clip_type = 'vit-base-patch16' #'vit-large-patch14-336'
            md_flag = "ViT-B/16" #"ViT-L/14@336px"
            is_image = False
            is_text = True
            neurons_to_fix1 = {}
        else:
            eval_data_type = 'fairface'
            data_type = 'fairface'
            clip_type = 'vit-large-patch14-336' #'vit-base-patch16'
            md_flag = "ViT-L/14@336px" #"ViT-B/16"
            is_image = True
            is_text = False
            neurons_to_fix1 = {98:0, 195:0}
            neurons_to_fix2 = {105:0, 163:0, 102:0}
            neurons_to_fix3 = {6:0, 264:0, 279:0, 98:0, 454:0, 364:0}
            layer = 23
    
            
        # llava = Llava() 
        sae_path = f"/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/checkpoints_dir/matroyshka_batch_top_k_20_x{expansion_factor}/random_k_2/{data_type}_train_activations_clip-{clip_type}_{layer}_post_mlp_residual_matroyshka_batch_top_k_20_x{expansion_factor}/trainer_0/checkpoints/ae_{epoch}.pt"
        sae = MatroyshkaBatchTopKSAE.from_pretrained(sae_path).cuda()

        # text = "Describe thie image in a single sentence." # Is this person baker or chef? Choose only one."
        # url = "/workspace/cvml_user/namin/bias_vlm/data/PAIRS/data/occupations/apron/white_man.png"
        # # image = Image.open(url) #requests.get(url, stream=True).raw)
        # image = Image.new("RGB", (224, 224), color="white") 
        # for neuron in [35, 38, 67]: #range(65536): # [258, 34, 81]: #[258]: # Note: ID of the pencil neuron may change after retraining SAE
        #     for alpha in [-30, -20, -10, 0, 10, 20, 30, 50]: #[0, 1.001, 1.01, 1.1, 1.5]: #[0, 30, 40, 50]:
        #         print(f"neuron {neuron}, alpha {alpha}")
        #         llava.attach_and_fix(sae=sae, neurons_to_fix={neuron: alpha}, pre_zero=False)
        #         output = llava.prompt(text, image, max_tokens=100)[0]
        #         print(output)
        #     print("======================================================")
        #     print()

        # alpha = 100
        # save_dir = f"/workspace/cvml_user/namin/bias_vlm/neuron_outputs_{data_type}"
        # file_prefix = f"neurons_{data_type}_{clip_type}_{layer}_{start_idx}_{end_idx}_{epoch}_{expansion_factor}"
        # save_path = os.path.join(save_dir, f"{file_prefix}.json")

        # # # # # # if args.save or not os.path.exists(save_path):
        # # neurons_lst = list(range(start_idx, end_idx))
        # # # # save neurons
        # # all_embeddings = []
        # # for neuron in tqdm(neurons_lst):
        # #     dec, emb = sae_clip.save_neurons(llava,
        # #                                      prompt="Describe this image in a single word.", sae=sae, neuron=neuron, alpha=alpha,
        # #                                      save_path=save_path)
        # #     # print(dec)
        # #     all_embeddings.append(emb)
        # # all_embeddings = torch.stack(all_embeddings, dim=0)
        # # save_emb_path = os.path.join(save_dir, f"embs_{file_prefix}.pt")
        # # torch.save(all_embeddings, save_emb_path)

        # file_name = "laion_400_unigram.txt" #"clip_disect_20k.txt"
        # file_path = f"/workspace/cvml_user/namin/bias_vlm/MSAE/vocab/{file_name}"  # Replace with your file path
        # with open(file_path, "r", encoding="utf-8") as f:
        #     words = [line.strip() for line in f if line.strip()]

        # # path = f"/workspace/cvml_user/namin/bias_vlm/neuron_outputs_{data_type}/neurons_0_{end_idx}_{epoch}_{expansion_factor}.json"
        # # words = []
        # # with open(path, "r") as f:
        # #     for line in f:
        # #         if not line.strip():
        # #             continue
        # #         data = json.loads(line)
        # #         response = data["response"].lower()
        # #         words.append(response)
        # # print(len(words))

        # save_dir = f"/workspace/cvml_user/namin/bias_vlm/neuron_outputs_{data_type}"
        # file_prefix = f"neurons_{start_idx}_{end_idx}_{epoch}_{expansion_factor}"

        # deb_clip_model, preprocess = sae_clip.load("ViT-B/16", device=device) #B/16", device=device) #L/14@336px", device=device)


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

        # # 1. Normalize vectors                                                          
        # vocab_embeddings = vocab_embeddings[:, 0, :]
        # embs = embs[:, 0, :]
        # embs = embs.to(vocab_embeddings.dtype)

        # decoder_norm = embs / embs.norm(dim=1, keepdim=True) # torch.Size([2048, 4096])
        # vocab_norm = vocab_embeddings / vocab_embeddings.norm(dim=1, keepdim=True) # torch.Size([37445, 768])


        # # 2. Compute cosine similarities
        # sim = decoder_norm @ vocab_norm.T  # [2048, 37445]

        # # 3. Get best match index for each neuron
        # best_idx = sim.argmax(dim=1)  # [2048]

        # # 4. Map back to vocab
        # neuron_to_vocab = [words[i] for i in best_idx.tolist()] # laion
        # print(set(neuron_to_vocab))

        # # else:
        # keywords = {
        #     "Woman": ['blondes'], # ["woman", "women", "female", "lady", "girl", "wife", "mother", "daughter"], # ["child", "baby"], #
        #     "Man": ['bearded', 'beards'] # ["man", "men", "male", "boy", "gentleman", "guy", "husband", "father", "son"] # ["old", "elderly"] #
        # }
        # alpha_man = 30# 100
        # alpha_woman = 30# 100
        
        # results = defaultdict(list)
        # for i,vocab in enumerate(neuron_to_vocab):
        #     for k in keywords.keys():
        #         if vocab in keywords[k]:
        #             results[k].append(i)
        # # with open(save_path, "r") as f:
        # #     for line in f:
        # #         if not line.strip():
        # #             continue
        # #         data = json.loads(line)
        # #         response = data["response"].lower()
        # #         neuron_id = data["neuron"]

        # #         # if response not in keywords.values():
        # #         #     results[response].append(neuron_id)

        # #         for category, words in keywords.items():
        # #             if any(word == response for word in words):
        # #                 results[category].append(neuron_id)
        
        # neurons_to_fix = {
        #     "Man": {neuron: alpha_man for neuron in results["Man"]},
        #     "Woman": {neuron: alpha_woman for neuron in results["Woman"]}
        # }

        # print(len(neurons_to_fix["Man"]))
        # print(len(neurons_to_fix["Woman"]))

        # man_neurons = neurons_to_fix["Man"]
        # woman_neurons = neurons_to_fix["Woman"] 
        # print(man_neurons)
        # print(woman_neurons)

        # # text = "Is this person a woman or man?"
        # # url = "/workspace/cvml_user/namin/bias_vlm/data/PAIRS/data/occupations/apron/white_man.png"
        # # image = Image.open(url) #requests.get(url, stream=True).raw)

        # # text = "Describe this image in a single sentence." #word."
        # # image = Image.new("RGB", (224, 224), color="white")  
        # # for neuron in [list(man_neurons.keys())[0], list(woman_neurons.keys())[0]]: # Note: ID of the pencil neuron may change after retraining SAE
        # #     temp_lst = []
        # #     for alpha in [30, 50]: #0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]:
        # #         llava.attach_and_fix(sae=sae, neurons_to_fix={neuron: alpha}, pre_zero=False)
        # #         output = llava.prompt(text, image, max_tokens=60)[0]
        # #         temp_lst.append(output)
        # #     print(temp_lst)
        # #     print("======================================================")

        # # # Balance count
        # min_len = min(len(man_neurons), len(woman_neurons))
        # print(min_len)
        # min_len = 1

        # # Randomly sample equal number of neurons
        # man_sample = dict(random.sample(man_neurons.items(), min_len)) # {i: alpha_man for i in random.sample(range(end_idx), min_len)} # dict(random.sample(man_neurons.items()), min_len)
        # woman_sample = dict(random.sample(woman_neurons.items(), min_len)) # {i: alpha_woman for i in random.sample(range(end_idx), min_len)} # dict(random.sample(woman_neurons.items()), min_len)

        # # Merge
        # neurons_to_fix_balanced = {**man_sample, **woman_sample}
        # deb_clip_model, preprocess = sae_clip.load("ViT-B/16", device=device) #L/14@336px", device=device)
        # deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix_balanced, pre_zero=False)

        # alpha = 10 #30 #1.5
        # deb_clip_model, preprocess = sae_clip.load("ViT-B/16", device=device, layer=layer)
        # print(sae)
        # print(sae.W_dec.shape)
        # print(sae.b_dec.shape)

        # w = sae.W_dec
        # torch.save(w, 'weight.pt')
        # b = sae.b_dec
        # print(b)
        # torch.save(b, 'bias.pt')
        
        # deb_clip_model, preprocess = sae_clip.load("ViT-L/14@336px", device=device, layer=layer)
        # print(deb_clip_model.debias_pos)
        # deb_clip_model.num_prompts_tokz = 0

        # ckpt = torch.load(os.path.join(args.ckpt_dir, 'linear_model_fairface.pt'))        

        # ig_save_path = os.path.join(args.ckpt_dir, f"integrated_gradients_{data_type}.pt") #_epoch_{epoch+1}.pt")
        # igs = torch.load(ig_save_path) # igs shape: (num_classes, num_data, num_neurons)
        # top_k = 10  # for example, top 10 neurons per class
        # igs_mean = igs.mean(dim=1)
        # _, topk_indices = torch.topk(igs_mean, k=top_k, dim=1) #, largest=False)
        # num_classes, num_neuron = topk_indices.shape



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
                    val, df = sae_clip.measure_bias(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, attribute=attribute,
                                                    prompt=prmpt, mod=args.mod, md_flag=args.md_flag, data_type=args.data_type,
                                                    is_siglip=is_siglip, is_blip=is_blip,
                                                    suppress_list=suppress_list, excite_dict=excite_dict,
                                                    sae=sae)
                    print(val)
                    # df.to_csv(f'summaries_{args.data_type}_{args.md_flag}_{attribute}_{prmpt}.csv')
        
        else:
            # neurons_to_fix = {4290:0, 3374:0, 648:0, 5052:0, 2277:0, 2887:0, 3896:0, 5458:0, 1321:0, 2327:0, 2675:0, 416:0, 5171:0, 1810:0, 4594:0, 1128:0, 1285:0, 4297:0} # random
            # neurons_to_fix = {105:0, 155:0, 340:0, 196:0, 335:0, 317:0, 749:0, 87:0, 266:0, 82:0, 85:0, 285:0, 696:0, 17:0, 163:0, 927:0, 337:0, 280:0, 84:0, 102:0, 36:0, 306:0, 276:0, 418:0, 124:0} # age
            # neurons_to_fix = {6:0, 831:0, 264:0, 135:0, 279:0, 358:0, 98:0, 155:0, 192:0, 328:0, 454:0, 364:0, 522:0} # race
            # neurons_to_fix = {98:0, 11:0, 124:0, 354:0, 203:0, 195:0, 236:0, 242:0, 254:0, 93:0, 733:0, 46:0, 302:0, 114:0, 117:0, 280:0, 331:0} # gender
            
            deb_clip_model, preprocess = sae_clip.load(md_flag, device=device, layer=layer)      
            try:
                deb_clip_model.eval()
            except:
                deb_clip_model.model.eval()
            for attribute in attribute_lst:
                for prmpt in prompts:
                    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                        deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix1, pre_zero=False, alpha=alpha, is_image=is_image)
                        val, df = sae_clip.measure_bias(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, attribute=attribute,
                                                    prompt=prmpt, mod=args.mod, md_flag=args.md_flag, data_type=eval_data_type,
                                                    is_siglip=is_siglip, is_blip=is_blip, epoch=epoch,
                                                    suppress_list=suppress_list, excite_dict=excite_dict,
                                                    sae=sae)
                        print(alpha, val)
            
            # if not args.text:
            #     deb_clip_model, preprocess = sae_clip.load(md_flag, device=device, layer=layer)
            #     try:
            #         deb_clip_model.eval()
            #     except:
            #         deb_clip_model.model.eval()
            #     for attribute in attribute_lst:
            #         for prmpt in prompts:
            #             for alpha in [1.0]: #, 1.0]: #[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            #                 deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix2, pre_zero=False, alpha=alpha, is_image=is_image)
            #                 val, df = sae_clip.measure_bias(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, attribute=attribute,
            #                                             prompt=prmpt, mod=args.mod, md_flag=args.md_flag, data_type=eval_data_type,
            #                                             is_siglip=is_siglip, is_blip=is_blip, epoch=epoch,
            #                                             suppress_list=suppress_list, excite_dict=excite_dict,
            #                                             sae=sae)
            #                 print(alpha, val)
                
            #     deb_clip_model, preprocess = sae_clip.load(md_flag, device=device, layer=layer)
            #     try:
            #         deb_clip_model.eval()
            #     except:
            #         deb_clip_model.model.eval()
            #     for attribute in attribute_lst:
            #         for prmpt in prompts:
            #             for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            #                 deb_clip_model.attach_and_fix(sae=sae, neurons_to_fix=neurons_to_fix3, pre_zero=False, alpha=alpha, is_image=is_image)
            #                 val, df = sae_clip.measure_bias(deb_clip_model, img_preproc=preprocess, tokenizer=tokenizer, attribute=attribute,
            #                                             prompt=prmpt, mod=args.mod, md_flag=args.md_flag, data_type=eval_data_type,
            #                                             is_siglip=is_siglip, is_blip=is_blip, epoch=epoch,
            #                                             suppress_list=suppress_list, excite_dict=excite_dict,
            #                                             sae=sae)
            #                 print(alpha, val)

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
    parser.add_argument("--ckpt_dir", type=str, default="/workspace/cvml_user/namin/bias_vlm/checkpoints")
    parser.add_argument("--text", action='store_true')
    parser.add_argument("--stereotype", action='store_true')

    args = parser.parse_args()
    main(args)