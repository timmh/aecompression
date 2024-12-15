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

1. `pip install requirements.txt`
2.  Download the iWildcam dataset here: [https://github.com/visipedia/iwildcam_comp](https://github.com/visipedia/iwildcam_comp). After unzipping, your directory structure should look as follows:

```
/your/path/to/iwildcam_unzipped/
  train/
  test/
  metadata/
```

3. Pre-train an autoencoder on the iWildcam training set:

```
python Iwildcam_Pretrain.py --dataset_root /your/path/to/iwildcam_unzipped/
```

Alternatively, you can download our pre-trained weights here: https://github.com/timmh/aecompression/releases/download/submission/8-epoch.04-val_loss.0.01.ckpt


3. Fine-tune on test locations using LoRA and QAT:

```
python lora_continual.py --dataset_root /your/path/to/iwildcam_unzipped/ --weights /path/to/your/pretrained_weights.ckpt
```

### Interactive demo

Turn your webcam into a camera trap! Our demo, `demo.ipynb` will capture a sequence of images from your webcam (much like a camera trap does). Take this moment to live out your fantasies of being a wild animal, roaming free in the wilderness, unaware of the human camera trap technology surveiling you. Take this guy as an example, caught as he emerges from behind a domesticated croton:

<img width="1206" alt="Screenshot 2024-12-14 at 7 22 49 PM" src="https://github.com/user-attachments/assets/e92ca293-d84f-4817-9e7a-e1ff9ab783da" />


Execute the notebook cells to run LoRA fine-tuning in real time, and observe the reconstruction getting better over time. 

<img width="1217" alt="Screenshot 2024-12-14 at 7 24 54 PM" src="https://github.com/user-attachments/assets/cedba39a-2709-424f-b7c3-9f8ae0d3280d" />


In a real-world deployment, we would submit the latent encoding of these images alongside the LoRA decoder weights, enabling low-bandwidth image transmission via satellite.
