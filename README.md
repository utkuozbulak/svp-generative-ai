# Improved Sub-Visible Particle Classification via Generative AI

This repository contains the official implementation, pretrained models, and sample datasets from the paper:

**Improved Sub-Visible Particle Classification in Flow Imaging Microscopy via Generative AI-Based Image Synthesis**  
Utku Ozbulak, Michaela Cohrs, Hristo L. Svilenov, Joris Vankerschaver, Wesley De Neve

## 🔬 Overview

Flow Imaging Microscopy (FIM) is a powerful tool for sub-visible particle (SvP) detection in biopharmaceuticals. However, data imbalance, particularly underrepresentation of particle types such as silicone oil and air bubbles, hinders the performance of multi-class deep learning classifiers.

In this work, we develop and validate a diffusion-based generative AI approach to synthesize realistic images of underrepresented particle types. We demonstrate that augmenting datasets with these synthetic images improves classification performance across multiple architectures.

## 📌 Key Contributions

- ✅ Diffusion models trained on 64×64 real FIM images of underrepresented particle types (air bubbles and silicone oil) that can generate high-fidelity, diverse synthetic samples to address data imbalance.
- ✅ Classificaiton models (ResNets) with > 95% Positive predictive value (PPV) for each class (protein particles, air bubbles, silicone oil) evaluated on 500k protein particles.
- ✅ Public release of all models and generated datasets.

### Visual Abstract for the Two Phases
<img src="https://raw.githubusercontent.com/utkuozbulak/svp-generative-ai/master/examples/Pharmacy_visual_abstract.png">

### Diffusion Process for AI-Generated SvPs
<img src="https://raw.githubusercontent.com/utkuozbulak/svp-generative-ai/master/examples/diffusion_process.png">

### Real (obtained from FIM) and AI-Generated SvP Images 
<img src="https://raw.githubusercontent.com/utkuozbulak/svp-generative-ai/master/examples/real_generated.png">

## 🛠 Repository Structure

This repository is organized as follows:
* **Code** - *src/* folder contains two subfolders: diffusion_code and classification_code, which contains code for their respective models

*src/diffusion_code* contains code to load and generate images from the difussion model

*src/classification_code* contains code to load classification models

* **Data** - *data/* folder contains a txt file with the appropriate link to the generated data, you can download this data and use it.

## :card_index_dividers: Models

To receive the weights of the trained models, please send an email to utku.ozbulak@ghent.ac.kr with a brief description of your intended use.

## :bar_chart: Data

Generated data are available at: https://zenodo.org/records/16757225

## 📝 License

This code, models, and dataset are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**BY** – Attribution: You must give credit to the original creator.  
**NC** – NonCommercial: You can’t use it for commercial purposes.  
**SA** – ShareAlike: If you remix or adapt it, you must share it under the same license.

This license applies to all components of this repository, including the trained models and the generated data, which permits non-commercial use, requires proper attribution, and mandates that any derivatives be shared under the same terms.

This work is a research output from Ghent University, Belgium and Ghent University Global Campus, South Korea.

For commercial licensing inquiries, please contact Utku Ozbulak from utku.ozbulak@ghent.ac.kr

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 📖 Citation

If you find the code in this repository useful for your research, consider citing our paper. Also, feel free to use any visuals available here.

    @misc{ozbulak2025improvedsubvisibleparticleclassification,
          title={Improved Sub-Visible Particle Classification in Flow Imaging Microscopy via Generative AI-Based Image Synthesis}, 
          author={Utku Ozbulak and Michaela Cohrs and Hristo L. Svilenov and Joris Vankerschaver and Wesley De Neve},
          year={2025},
          eprint={2508.06021},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2508.06021}, 
    }
