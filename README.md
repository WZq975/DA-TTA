# DA-TTA: Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams

This is the **official implementation** of the paper:  
**"Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams"**  
*Presented at ECCV 2024.*  

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2407.12128)

---

## 🏃 Run Instructions

To run the implementation, use the following command:  
```bash
python tta.py --cfg cfgs/[dataset, e.g., cifar100_c]/[method, e.g., da-tta].yaml
```
Replace `[dataset]` and `[method]` with your desired dataset and method configuration.

---

## 📂 Dataset Downloads

The following datasets are supported and can be downloaded from the provided links:

- **CIFAR10-C**: [Download here](https://zenodo.org/records/2535967#.ZBiI7NDMKUk)
- **CIFAR100-C**: [Download here](https://zenodo.org/records/3555552#.ZBiJA9DMKUk)
- **ImageNet-C**: [Download here](https://zenodo.org/records/2235448#.Yj2RO_co_mF)
- **ImageNet-D**: [GitHub Link](https://github.com/bethgelab/robustness/tree/main/examples/imagenet_d)
- **ImageNet-R**: [GitHub Link](https://github.com/hendrycks/imagenet-r)

---

## 📝 Citation

```bibtex
@inproceedings{wang2025datta,
  title={Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams},
  author={Wang, Ziqiang and Chi, Zhixiang and Wu, Yanan and Gu, Li and Liu, Zhi and Plataniotis, Konstantinos and Wang, Yang},
  booktitle={European Conference on Computer Vision},
  pages={332--349},
  year={2024},
  organization={Springer}
}
```
---

## 🙏 Acknowledgements
The benchmark framework (data loading, SOTA methods, etc.) is from the [online test-time adaptation repository](https://github.com/mariodoebler/test-time-adaptation).
