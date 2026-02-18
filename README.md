# Anatomical Variation Modeling for Radiation Therapy

This repository contains **data and code** for training and testing generative models that predict anatomical variations across radiation therapy fractions, along with their impact on 3D dose distributions.

---

## 📋 Repository Overview

This project develops generative models to learn realistic anatomical changes that occur during multi-fraction radiation therapy. The models are conditioned on first fraction CT scans and generate plausible anatomical variants that match the distribution of real anatomical changes observed in subsequent treatment fractions.

---

## 📂 Repository Structure

```
.
├── code
│     ├── ddf/                          # Dense Deformation Field approach
│     ├── stable_diffusion/             # Stable Diffusion approach  
│     ├── train_test_data_split.py      # Data preparation script
│     ├── data_dict.json                # Train/test/fold assignments
│     └── README.md                     # readme
├── data
│     ├── CT/                           # Folder with patient subfolders, each subfolder containing fraction CTs in nifty format 
│     ├── DOSE/                         # Folder with patient subfolders, each subfolder containing planned dose in nifty format
│     ├── STRUCTURES/                   # Folder with patient subfolders, each subfolder containing segmented OARs and target
│     ├── prepareFractionCT.py          # Script for converting DICOMS to nifty
│     ├── prostaty.txt                  # TXT file with target names used in our database
│     └── README.md                     # readme
├── LICENSE                            # License statement
└── README.md                          # This file

```

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- SimpleITK
- PyTorch

---

## 🔑 Applications

This framework enables:

1. **Robust treatment planning**: Account for anatomical variations in dose optimization
2. **Adaptive radiotherapy**: Predict anatomical changes for plan adaptation
3. **Uncertainty quantification**: Quantify dose uncertainty due to anatomical variations
4. **Quality assurance**: Validate delivered dose against predicted variations
5. **Research**: Study inter-fraction anatomical changes and their dosimetric impact

---

## 📖 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{your_paper,
  title={Generative Models for Anatomical Variation in Radiation Therapy},
  author={},
  journal={},
  year={2026},
  doi={}
}
```

---

## 📝 License

MIT

---

## ⚠️ Data Privacy

This repository structure is designed for research purposes. When working with patient data:
- Ensure all data is **de-identified** and compliant with HIPAA/GDPR
- Obtain appropriate **IRB approval** for your institution
- Follow institutional data sharing and privacy policies
- Do not include identifiable patient information in public repositories
- **DICOM files must be anonymized** before conversion to NIfTI

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional generative model architectures
- Enhanced uncertainty quantification methods
- Integration with treatment planning systems
- Validation on additional datasets
- Improved DICOM parsing for different vendors

Please open an issue or submit a Pull Request.

---

## 📧 Contact

For questions or collaborations, please contact ztabor@agh.edu.pl
