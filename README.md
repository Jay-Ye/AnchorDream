# AnchorDream

This is the repository for the [AnchorDream project website](https://jay-ye.github.io/AnchorDream).

**AnchorDream: Repurposing Video Diffusion for Embodiment-Aware Robot Data Synthesis**

[Junjie Ye](https://jay-ye.github.io/)<sup>1,2</sup>, [Rong Xue](https://rongxuezoe.github.io/)<sup>2</sup>, [Basile Van Hoorick](https://basile.be/)<sup>1</sup>, [Pavel Tokmakov](https://pvtokmakov.github.io/home/)<sup>1</sup>, [Muhammad Zubair Irshad](https://zubairirshad.com/)<sup>1</sup>, [Yue Wang](https://yuewang.xyz/)<sup>2</sup>, [Vitor Guizilini](https://vitorguizilini.github.io/)<sup>1</sup>

<sup>1</sup>Toyota Research Institute &nbsp;&nbsp; <sup>2</sup>USC Physical Superintelligence (PSI) Lab

**ICRA 2026**

[Paper](https://arxiv.org/pdf/2512.11797) | [arXiv](https://arxiv.org/abs/2512.11797) | [Code](https://github.com/Jay-Ye/AnchorDream)

## Abstract

The collection of large-scale and diverse robot demonstrations remains a major bottleneck for imitation learning, as real-world data acquisition is costly and simulators offer limited diversity and fidelity. To address these limitations, we introduce **AnchorDream**, an embodiment-aware world model that repurposes pretrained video diffusion models for robot data synthesis. AnchorDream conditions the diffusion process on robot motion renderings, anchoring the embodiment to prevent hallucination while synthesizing objects and environments consistent with the robot's kinematics. Starting from only a handful of human teleoperation demonstrations, our method scales them into large, diverse, high-quality datasets without requiring explicit environment modeling. Experiments show that the generated data leads to consistent improvements in downstream policy learning, with relative gains of **36.4% in simulator benchmarks** and nearly **double performance in real-world studies**.

## Citation

If you find AnchorDream useful for your work please cite:
```
@inproceedings{ye2025anchordream,
  title={AnchorDream: Repurposing Video Diffusion for Embodiment-Aware Robot Data Synthesis},
  author={Ye, Junjie and Xue, Rong and Van Hoorick, Basile and Tokmakov, Pavel and Irshad, Muhammad Zubair and Wang, Yue and Guizilini, Vitor},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

The website template is borrowed from [Nerfies](https://github.com/nerfies/nerfies.github.io).
