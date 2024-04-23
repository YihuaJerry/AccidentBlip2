# AccidentBlip2: Accident Detection With Multi-View MotionBlip2
Official Implementation for AccidentBlip2. [Arxiv](https://arxiv.org/pdf/2404.12149.pdf)

## Abstract
Intelligent vehicles have demonstrated excellent capabilities in many transportation scenarios, but the complex on-board sensors and the inference capabilities of on-board neural networks limit the accuracy of intelligent vehicles for accident detection in complex transportation systems. In this paper, we present AccidentBlip2, a pure vision-based multimodal large model Blip2 accident detection method. Our method first processes the multi-view through ViT-14g and inputs the multi-view features into the cross attention layer of the Qformer, while our self-designed Motion Qformer replaces the self-attention layer in Blip2's Qformer with the Temporal Attention layer in the In the inference process, the query generated in the previous frame is input into the Temporal Attention layer to realize the inference for temporal information. Then we detect whether there is an accident in the surrounding environment by performing autoregressive inference on the query input to the MLP. We also extend our approach to a multi-vehicle cooperative system by deploying Motion Qformer on each vehicle and simultaneously inputting the inference-generated query into the MLP for autoregressive inference. Our approach detects the accuracy of existing video large language models and also adapts to multi-vehicle systems, making it more applicable to intelligent transportation scenarios.


## Installation
- install `transformers` and `torch` library beforehand
  
  ```bash
  pip3 install transformers torch
  ```
- Replace `./blip2/modeling_blip2.py` into your own transformers environment
- Run the training code
