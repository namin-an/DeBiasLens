# Interpretable Debiasing of Vision-Language Models for Social Fairness (CVPR 2026)

<br>

 <a href="https://arxiv.org/abs/2602.24014"><img src="https://img.shields.io/badge/Paper-arXiv:2602.24014-red"></a>

<br>

Authors: [Na Min An](https://namin-an.github.io/)
[Yoonna Jang](https://yoonnajang.github.io/)
[Yusuke Hirota](https://www.y-hirota.com/)
[Ryo Hachiuma](https://ryohachiuma.github.io/)
[Isabelle Augenstein](https://isabelleaugenstein.github.io/)
[Hyunjung Shim](https://kaist-cvml.github.io/)

<br>

<p align="center" width="100%"><img src="./assets/diagram.png" alt="DeBiasLens Overview"></img></p>   

<br>

## Key Contributions

- 🔍 **Interpretable Internals**: The first interpretable debiasing framework to discover neurons highly responsive to specific social attributes.
- ⚙️ **Model-Agnostic Debiasing**: Applicable across different VLM architectures without full model retraining.
- ✅ **Bias Mitigation without Capability Degradation**: Selective neuron deactivation preserves general semantic understanding.

---

## Installation

```bash
git clone https://github.com/<your-username>/debiaslens.git
cd debiaslens
pip install -r requirements.txt
```

---

## Usage

### 1. Train Sparse Autoencoders

Train SAEs on facial image or caption datasets to uncover latent social attribute neurons.

### 2. Localize Social Attribute Neurons

Identify neurons most responsive to specific demographic attributes.

### 3. Debiased Inference

Run VLM inference with targeted neuron deactivation.

---

## Repository Structure

```
debiaslens/
├── train_sae.py              # SAE training script
├── localize_neurons.py       # Neuron localization pipeline
├── inference_debias.py       # Debiased inference
├── models/
│   ├── sae.py                # Sparse Autoencoder architecture
│   └── vlm_wrapper.py        # Model-agnostic VLM interface
├── data/
│   └── README.md             # Dataset download instructions
├── evaluation/
│   ├── bias_metrics.py       # Bias evaluation metrics
│   └── semantic_metrics.py   # Semantic capability benchmarks
├── configs/                  # Experiment configuration files
├── assets/                   # Figures and visualizations
├── requirements.txt
└── README.md
```

---

## Citation
Please consider citing our work if you find this work helpful for your research.

```
@article{an2026interpretable,
  title={Interpretable Debiasing of Vision-Language Models for Social Fairness},
  author={An, Na Min and Jang, Yoonna and Hirota, Yusuke and Hachiuma, Ryo and Augenstein, Isabelle and Shim, Hyunjung},
  journal={arXiv preprint arXiv:2602.24014},
  year={2026}
}
```

---

## Acknowledgements

We thank the teams behind the open-source VLM and SAE libraries that made this work possible. This research was conducted in collaboration across KAIST, the University of Copenhagen, and NVIDIA.

---

## Contact
- Na Min An: naminan@kaist.ac.kr
- Hyunjung Shim: kateshim@kaist.ac.kr

