# Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping

Zijie Pan, [Jiachen Lu](https://victorllu.github.io/), [Xiatian Zhu](https://xiatian-zhu.github.io/), [Li Zhang](https://lzrobots.github.io/)

[[`Paper`]()] [[`Project`](https://github.com/fudan-zvg/PGC-3D)] [[`BibTeX`](#citing-pgc-3d)]

https://github.com/fudan-zvg/PGC-3D/assets/84657631/132a1379-47d1-4213-9da6-03084bb4f8b6

**PGC** (Pixel-wise Gradient Clipping) introduces a refined method to adapt traditional gradient clipping. By focusing on pixel-wise gradient magnitudes, it retains vital texture details. This approach acts as a versatile **plug-in**, seamlessly complementing existing **SDS and LDM-based 3D generative models**. The result is a marked improvement in high-resolution 3D texture synthesis.

With PGC, users can:
- Address and mitigate gradient-related challenges common in LDM, elevating the quality of 3D generation.
- Employ the [**SDXL**](https://github.com/Stability-AI/generative-models) approach, previously not adaptable for 3D generation.

## Results
Incorporating the proficient and potent **PGC** implementation into SDXL guidance has led to notable advancements in 3D generation results.

#### [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)
<img width="4096" alt="photo" src="assets/group_photo_fan.png">

#### Ours
<img width="4096" alt="photo" src="assets/group_photo.png">

https://github.com/fudan-zvg/PGC-3D/assets/84657631/462c8e5e-e2f6-4709-b864-379413df55ad

## Tips for SDXL
- Using [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- [Controlnet](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0) may be important to make training easier to converge

## Citing PGC-3D
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:
```
@article{karaev2023cotracker,
  title={Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping},
  author={Pan, Zijie and Lu, Jiachen and Zhu, Xiatian and Zhang, Li},
  journal={arXiv},
  year={2023}
}
```