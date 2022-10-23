# Autoregressive GAN for Semantic Unconditional Head Motion Generation

## Abstract

We address the task of unconditional head motion generation in order to animate still human faces in a low dimensional semantic space.
Deviating from talking head generation conditioned on audio that seldom put emphasis on realistic head motions, we identify key components in a GAN-based architecture that allow to obtain rich head motion sequences while avoiding known caveats associated with GANs.
Namely, autoregressive generation of incremental outputs ensures smooth trajectories, while a multi-scale discriminator on input pairs drives generation towards better handling of high and low frequency signals and less mode collapse.
We demonstrate experimentally the relevance of the proposed architecture and compare with models that showed state-of-the-art performance on similar tasks. 

## Architecture overview

![uncond_head_mot](https://user-images.githubusercontent.com/36541517/197400808-c6094353-4bb7-4e49-8dd8-8f325aa4539a.png)

## Demo

![dem_rnn_cf_8_2_colors](https://user-images.githubusercontent.com/36541517/197400986-43a91f7f-369d-4ea4-b2d8-664d592f65c7.gif)
