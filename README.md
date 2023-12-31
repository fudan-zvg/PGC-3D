# Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping
### [[Paper]](https://arxiv.org/abs/2310.12474) | [[Project]](https://fudan-zvg.github.io/PGC-3D/)

> [**Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping**](https://arxiv.org/abs/2310.12474),            
> Zijie Pan, [Jiachen Lu](https://victorllu.github.io/), [Xiatian Zhu](https://surrey-uplab.github.io/), [Li Zhang](https://lzrobots.github.io)  
> **Arxiv preprint**

**Official implementation of "Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping".** 


**PGC** (Pixel-wise Gradient Clipping) introduces a refined method to adapt traditional gradient clipping. By focusing on pixel-wise gradient magnitudes, it retains vital texture details. This approach acts as a versatile **plug-in**, seamlessly complementing existing **SDS and LDM-based 3D generative models**. The result is a marked improvement in high-resolution 3D texture synthesis.

With PGC, users can:
- Address and mitigate gradient-related challenges common in LDM, elevating the quality of 3D generation.
- Employ the [**SDXL**](https://github.com/Stability-AI/generative-models) approach, previously not adaptable for 3D generation.

## News
- We have tested the environment and common usage. This repo will offer an unified implementation for mesh optimization and reproduction of many SDS variants. Codes will be released soon.

## Results
Incorporating the proficient and potent **PGC** implementation into SDXL guidance has led to notable advancements in 3D generation results.

#### [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)
<img width="4096" alt="photo" src="assets/group_photo_fan.png">

#### Ours
<img width="4096" alt="photo" src="assets/group_photo.png">

https://github.com/fudan-zvg/PGC-3D/assets/84657631/a04cd157-6666-4fd3-b3a1-069f4ec8f255

## Tips for SDXL
- Using [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- [Controlnet](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0) may be important to make training easier to converge

## BibTeX
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:
```
@article{pan2023enhancing,
  title={Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping},
  author={Pan, Zijie and Lu, Jiachen and Zhu, Xiatian and Zhang, Li},
  journal={arXiv preprint arXiv 2310.12474},
  year={2023}
}
```
