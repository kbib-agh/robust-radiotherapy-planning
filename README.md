# Generative Models for Anatomical Variations

This repository contains code for **training and testing generative models** that learn anatomical variations from CT scans across radiation therapy fractions.

---

## 📋 Overview

The project focuses on generating **plausible anatomical variants** using CT scans from different fractions of radiation therapy. The models learn to predict realistic anatomical changes conditioned on the first fraction CT scan.

---

## 📂 Repository Structure

```
.
├── code
     ├── ddf/                          # Dense Deformation Field approach
     ├── stable_diffusion/             # Stable Diffusion approach  
     ├── train_test_data_split.py      # Data preparation script
     ├── data_dict.json                # Train/test/fold assignments
     └── README.md                     # readme
├── data
     ├── CT/                           # Folder with patient subfolders, each subfolder containing fraction CTs in nifty format 
     ├── DOSE/                         # Folder with patient subfolders, each subfolder containing planned dose in nifty format
     ├── STRUCTURES/                   # Folder with patient subfolders, each subfolder containing segmented OARs and target
     ├── prepareFractionCT.py          # Script for converting DICOMS to nifty
     ├── prostaty.txt                  # TXT file with target names used in our database
     └── README.md                     # readme
├── LICENSE                            # License statement
└── README.md                          # This file

```

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- SimpleITK
- PyTorch
- Additional dependencies (see `requirements.txt`)

---

## 📖 Citation

If you use this code in your research, please cite our work:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

---

## 📝 License

[Specify your license here]

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or collaborations, please contact [your email].
