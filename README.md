# Skyreels V1: Human-Centric Video Foundation Model

<p align="center">
🤗 <a href="https://huggingface.co/Skywork" target="_blank">Hugging Face</a> · <a href="https://www.skyreels.ai/" target="_blank">Playground</a>
</p>

---
Welcome to the Skyreels V1 repository! Here, you'll find the Text-to-Video & Image-to-Video model weights and inference code for our groundbreaking video foundation model.

## Overview

Skyreels V1 is the first and most advanced open-source Human-Centric Video Foundation Model. It is trained on over 100 million high-quality film and television datasets, leveraging the <a href="https://huggingface.co/tencent/HunyuanVideo"> HunyuanVideo</a> model. Skyreels V1 highlights:

1. **Leading Open-Source Model**: Our Image-to-Video model achieves state-of-the-art (SOTA) performance among open-source models, comparable to proprietary models like kling and hailuo.
2. **Facial Expression Support**: Captures 162 distinct facial expressions and over 400 natural movement combinations, accurately reflecting human emotions.
3. **Camera Movement Templates**: Includes 14 industry-standard camera movement templates.

## Key Features

Our model is built on a self-developed data cleaning and annotation pipeline, creating a vast dataset of high-quality film, television, and documentary content. Key features include:

- **Expression Classification**: Categorizes human facial expressions into 54 distinct types.
- **Character Spatial Awareness**: Utilizes 3D human reconstruction technology to understand spatial relationships between multiple people in a video, enabling film-level character positioning.
- **Action Recognition**: Identifies over 400 action semantic units with a recognition accuracy of over 90%.
- **Scene Understanding**: Conducts cross-modal correlation analysis of clothing, scenes, and plots.


## Benchmark Results
We evaluate the performance of our model using <a href="https://github.com/Vchitect/VBench">Vbench</a>, comparing with other outperformance open-source models.
| Model              | Total Score | I2V Score | Quality Score |
|--------------------|-------------|-----------|---------------|
| DynamiCrafter      | 1.2         | 1.5       | 1.8           |
| CogvideoX 1.5      | 1.3         | 1.6       | 1.9           |
| Skyreels V1        | 1.4         | 1.7       | 2.0           |


## Demos

## Model Introduction
| Model Name      | Resolution | Video Length | FPS | Download Link |
|-----------------|------------|--------------|-----|---------------|
| Skyreels V1 I2V | 544px960p  | 97           | 24  | [Download](https://skyreels-v1-i2v.oss-cn-shanghai.aliyuncs.com/skyreels_v1_i2v.zip) |
| Skyreels V1 T2V | 544px960p  | 97           | 24  | [Download](https://skyreels-v1-t2v.oss-cn-shanghai.aliyuncs.com/skyreels_v1_t2v.zip) |


## Usage
