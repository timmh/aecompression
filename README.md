## On-Device Parameter-Efficient Fine-Tuning for Edge-Enabled Wildlife Camera Traps
### Final Project for 6.5940 TinyML and Efficient Deep Learning Computing, Fall 2024
#### Timm Haucke and Justin Kay

Automatically triggered cameras, also known as camera traps, are an indispensable tool for studying animal ecology. 
Retrieving data from remote camera traps is an ongoing challenge. Due to bandwidth limitations, it is not practical to utilize satellite transmission in most cases. 
We propose a method for efficient deep-learning based image compression to address this challenge. Our method utilizes a state-of-the-art autoencoder network to compress 
images down into a small latent space of between 8 to 256 dimensions. By simply transmitting these latent representations we demonstrate that in many cases we can reduce 
bandwidth compared to JPEG while achieving the same or better reconstruction quality. We also propose a method for deployment-specific fine-tuning of our autoencoder 
architecture to specialize models at the edge to specific environmental conditions. Specifically, we fine-tune the decoder with LoRA and quantization-aware training, 
and transmit the quantized LoRA weights as well. Our experiments demonstrate that this approach incurs limited additional bandwidth requirements and improves reconstruction error, 
particularly at locations with abundant imagery.

### Reproducing our experiments

1. Download the iWildcam dataset here: https://github.com/visipedia/iwildcam_comp

2. Pre-train an autoencoder on the iWildcam training set:

```
python Iwildcam_Pretrain.py
```

3. Fine-tune on test locations using LoRA and QAT:

```
python lora_continual.py
```

### Interactive demo

