#!/bin/bash


# nohup python ../evaluate.py --md_flag clip-vitb16 &> nohup_clip-vitb16_stereo.out

# nohup python ../evaluate.py --md_flag debias_clip &> nohup_debclip.out

# nohup python ../evaluate.py --md_flag clip-vitl14 &> nohup_clip-vitl14_stereo.out

# nohup python ../evaluate.py --mod vision --neuron &> nohup_mmneuron.out

# nohup python ../evaluate.py --md_flag sae_clip --text &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_final_text.out"

# for data in fairface #celeba #fairface cocogender celeba
# do
#     for epoch in 100000 #0 20000 40000 60000 80000 100000
#     do
#         echo "Running evaluation for prompt type: $datat and epoch $epoch" 
#         nohup python ../evaluate.py --md_flag sae_clip --data_type "$data" --epoch "$epoch" &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_final.out"
#     done
# done


nohup python ../evaluate_final_both.py --md_flag sae_clip --stereotype &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_stereotype_final_both_large.out"

# nohup python ../evaluate_final_both.py --md_flag sae_clip &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_final_both_large.out"
# nohup python ../evaluate_final_both.py --md_flag sae_clip &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_large_final.out"


# nohup python ../evaluate_final_both.py --md_flag sae_clip --stereotype &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_stereotype_final_both_raceneurons.out"
# nohup python ../evaluate_final.py --md_flag sae_clip --stereotype &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_stereotype_large_final.out"


# nohup python ../evaluate.py --md_flag sae_clip --text --data_type pata &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_pata_final_text.out"

# for data in fairface #celeba #fairface cocogender celeba
# do
#     for epoch in 100000 #0 20000 40000 60000 80000 100000
#     do
#         echo "Running evaluation for prompt type: $datat and epoch $epoch" 
#         nohup python ../evaluate.py --md_flag sae_clip --data_type "$data" --epoch "$epoch" --data_type pata &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_pata_final.out"
#     done
# done


# nohup python ../evaluate.py --md_flag sae_clip --text --stereotype &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_stereotype_final_text.out"

# for data in fairface #celeba #fairface cocogender celeba
# do
#     for epoch in 100000 #0 20000 40000 60000 80000 100000
#     do
#         echo "Running evaluation for prompt type: $datat and epoch $epoch" 
#         nohup python ../evaluate.py --md_flag sae_clip --data_type "$data" --epoch "$epoch" --stereotype &> "nohup_sae_clip_vitb16${data}_epoch${epoch}_stereotype_final.out"
#     done
# done






# nohup python ../evaluate.py --md_flag debias_clip --data_type pata &> nohup_debias_clip_pata.out

# nohup python ../evaluate.py --md_flag clip-vitb16 --data_type pata &> nohup_clip-vitb16_pata.out

# nohup python ../evaluate.py --md_flag sae_clip --data_type pata &> nohup_sae_clip_pata.out


# nohup python ../evaluate.py --md_flag clip-vitb32 &> nohup_clip-vitb32.out


# nohup python ../evaluate.py --md_flag siglip &> nohup_siglip.out


# nohup python ../evaluate.py --md_flag blip-itm &> nohup_blip.out


# nohup python ../evaluate.py --md_flag clip-vitl14 --data_type pairs &> nohup_clip-vitl14_pairs.out


# nohup python ../evaluate.py --mod vision --neuron --data_type pairs &> nohup_pairs.out


# nohup python ../evaluate.py --mod vision --neuron --data_type pata &> nohup_pata.out

