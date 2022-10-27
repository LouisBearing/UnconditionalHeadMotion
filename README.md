# Autoregressive GAN for Semantic Unconditional Head Motion Generation (SUHMo)

## Abstract

We address the task of unconditional head motion generation to animate still human faces in a low-dimensional semantic space.
Deviating from talking head generation conditioned on audio that seldom puts emphasis on realistic head motions, we devise a GAN-based architecture that allows obtaining rich head motion sequences while avoiding known caveats associated with GANs.
Namely, the autoregressive generation of incremental outputs ensures smooth trajectories, while a multi-scale discriminator on input pairs drives generation toward better handling of high and low frequency signals and less mode collapse.
We demonstrate experimentally the relevance of the proposed architecture and compare with models that showed state-of-the-art performances on similar tasks. 

## Architecture overview

![uncond_head_mot](https://user-images.githubusercontent.com/36541517/197400808-c6094353-4bb7-4e49-8dd8-8f325aa4539a.png)

## Demo

1 reference pose --> 120 predicted frames

### SUHMo RNN - Training on CONFER DB

<img src="media/git_demo_vis/rnn/dem_rnn_cf_8_16_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_9_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_1_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_22_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_24_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_27_colors.gif" alt="drawing" width="200"/>
<img src="media/git_demo_vis/rnn/dem_rnn_cf_8_37_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_41_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_45_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_49_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_51_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_6_colors.gif" alt="drawing" width="200"/>

### SUHMo RNN - Training on VoxCeleb2

<img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_7_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_22_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_24_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_26_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_28_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_29_colors.gif" alt="drawing" width="200"/>
<img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_30_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_31_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_38_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_39_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_40_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_7_colors.gif" alt="drawing" width="200"/>

### Comparison with ACTOR model

#### Training on VoxCeleb2
<img src="media/git_demo_vis/actor_cf/dem_actor_vox4work_0_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_cf/dem_actor_vox4work_17_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_cf/dem_actor_vox4work_25_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_cf/dem_actor_vox4work_40_colors.gif" alt="drawing" width="200"/>
