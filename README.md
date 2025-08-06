# Improved Sub-Visible Particle Classification via Generative AI

This repository contains the official implementation, pretrained models, and sample datasets from the paper:

**Improved Sub-Visible Particle Classification in Flow Imaging Microscopy via Generative AI-Based Image Synthesis**  
Utku Ozbulak\*, Michaela Cohrs, Hristo L. Svilenov, Joris Vankerschaver, Wesley De Neve

For queries, please reach out to: utku.ozbulak@ghent.ac.kr

## 🔬 Overview

Flow Imaging Microscopy (FIM) is a powerful tool for sub-visible particle (SvP) detection in biopharmaceuticals. However, data imbalance, particularly underrepresentation of particle types such as silicone oil and air bubbles, hinders the performance of multi-class deep learning classifiers.

In this work, we develop and validate a diffusion-based generative AI approach to synthesize realistic images of underrepresented particle types. We demonstrate that augmenting datasets with these synthetic images improves classification performance across multiple architectures.

## 📌 Key Contributions

- ✅ A **diffusion model** trained on 64x64 real FIM images of underrepresented particle types (air bubbles and silicone oil).
- ✅ Multi-class **deep neural network classifiers** (ResNet-18 and ResNet-50) trained on real and augmented datasets with state-of-the-art classification accuracy on 3 class classification (protein particles, air bubbles, silicone oil).
- ✅ Public release of all models and generated datasets.

---

### Example Diffusion Process for SvPs
<img src="https://raw.githubusercontent.com/utkuozbulak/svp-generative-ai/master/examples/diffusion_process.png">

### Example Images 
<img src="https://raw.githubusercontent.com/utkuozbulak/svp-generative-ai/master/examples/real_generated.png">

---

## 🛠 Repository Structure

This repository is organized as follows:
* **Code** - *src/* folder contains two subfolders: diffusion_code and classification_code, which contains code for their respective models

*src/diffusion_code* contains code to load and generate images from the difussion model

*src/classification_code* contains code to load classification models

* **Data** - *data/* folder contains a txt file with the appropriate link to the generated data, you can download this data and use it.

## :card_index_dividers: Models

To receive the weights of the trained models, please send an email to utku.ozbulak@ghent.ac.kr with a brief description of your intended use.

## :bar_chart: Data

Generated data are available at: ....

## 📝 License

This code, models, and dataset are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**BY** – Attribution: You must give credit to the original creator.  
**NC** – NonCommercial: You can’t use it for commercial purposes.  
**SA** – ShareAlike: If you remix or adapt it, you must share it under the same license.

This license applies to all components of this repository, including the trained models and the generated data, which permits non-commercial use, requires proper attribution, and mandates that any derivatives be shared under the same terms.

---

This work is a research output from Ghent University, Belgium and Ghent University Global Campus, South Korea.

For commercial licensing inquiries, please contact Utku Ozbulak

EUR: utku.ozbulak@ugent.be - Korea/US: utku.ozbulak@ghent.ac.kr

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
