# Autoregressive GAN for Semantic Unconditional Head Motion Generation (SUHMo)

## Abstract

We address the task of unconditional head motion generation to animate still human faces in a low-dimensional semantic space.
Deviating from talking head generation conditioned on audio that seldom puts emphasis on realistic head motions, we devise a GAN-based architecture that allows obtaining rich head motion sequences while avoiding known caveats associated with GANs.
Namely, the autoregressive generation of incremental outputs ensures smooth trajectories, while a multi-scale discriminator on input pairs drives generation toward better handling of high and low frequency signals and less mode collapse.
We demonstrate experimentally the relevance of the proposed architecture and compare with models that showed state-of-the-art performances on similar tasks. 

## Architecture overview

![uncond_head_mot](https://user-images.githubusercontent.com/36541517/197400808-c6094353-4bb7-4e49-8dd8-8f325aa4539a.png)

## Demo

In the results presented below **120 frames** are generated from a **single** reference image.

### SUHMo-RNN (Training on CONFER DB)
<p align="center">
<img src="media/git_demo_vis/rnn/dem_rnn_cf_8_9_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_2_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_1_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_22_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_24_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_27_colors.gif" alt="drawing" width="200"/>
<img src="media/git_demo_vis/rnn/dem_rnn_cf_8_37_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_41_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_45_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_49_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_51_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/rnn/dem_rnn_cf_8_6_colors.gif" alt="drawing" width="200"/>
 </p>

### SUHMo-Transformer (Training on VoxCeleb2)

<p align="center">
<img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_7_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_22_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_24_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_26_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_28_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_29_colors.gif" alt="drawing" width="200"/>
<img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_30_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_31_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_38_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_39_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_40_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/transfo/dem_transf_vox_sub0.1_dec1050_1_7_colors.gif" alt="drawing" width="200"/>
</p>

<!-- ### Comparison with ACTOR model

#### Training on CONFER DB
<p align="center">
<img src="media/git_demo_vis/actor_cf/dem_actor_0_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_cf/dem_actor_17_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_cf/dem_actor_25_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_cf/dem_actor_40_colors.gif" alt="drawing" width="200"/>
  </p>

#### Training on VoxCeleb2
<p align="center">
<img src="media/git_demo_vis/actor_vox/dem_actor_21_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_vox/dem_actor_vox4work_22_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_vox/dem_actor_vox4work_3_colors.gif" alt="drawing" width="200"/> <img src="media/git_demo_vis/actor_vox/dem_actor_vox4work_57_colors.gif" alt="drawing" width="200"/>
  </p> -->

### SUHMo in-the-wild

Several outputs can be obtained from the same reference image. See below for an illustration on SUHMo-RNN trained on CONFER DB.

<p align="center"><img src="media/demo_img/Legend.png" alt="drawing" width="500"/></p>

<img src="media/demo_img/audrey.png" alt="drawing" width="180"/><img src="media/demo_img/img2seq_rnn_1/moving_audrey.gif" alt="drawing" width="126"/><img src="media/demo_img/img2seq_rnn_2/moving_audrey.gif" alt="drawing" width="126"/><img src="media/demo_img/captain.png" alt="drawing" width="180"/><img src="media/demo_img/img2seq_rnn_1/moving_captain2.gif" alt="drawing" width="126"/><img src="media/demo_img/img2seq_rnn_2/moving_captain2.gif" alt="drawing" width="126"/>

<p align="center">
<img src="media/demo_img/cesi.png" alt="drawing" width="200"/><img src="media/demo_img/img2seq_rnn_1/moving_cesi.gif" alt="drawing" width="140"/><img src="media/demo_img/img2seq_rnn_2/moving_cesi.gif" alt="drawing" width="140"/><img src="media/demo_img/jon.png" alt="drawing" width="200"/><img src="media/demo_img/img2seq_rnn_1/moving_john.gif" alt="drawing" width="140"/><img src="media/demo_img/img2seq_rnn_2/moving_john.gif" alt="drawing" width="140"/>
</p>
 
<p align="center">
<img src="media/demo_img/monalisa.png" alt="drawing" width="200"/><img src="media/demo_img/img2seq_rnn_1/moving_monalisa2.gif" alt="drawing" width="140"/><img src="media/demo_img/img2seq_rnn_2/moving_monalisa2.gif" alt="drawing" width="140"/><img src="media/demo_img/morgan.png" alt="drawing" width="200"/><img src="media/demo_img/img2seq_rnn_1/moving_morgan.gif" alt="drawing" width="140"/><img src="media/demo_img/img2seq_rnn_2/moving_morgan.gif" alt="drawing" width="140"/>
</p>

<p align="center">
<img src="media/demo_img/paint.png" alt="drawing" width="200"/><img src="media/demo_img/img2seq_rnn_1/moving_paint1.gif" alt="drawing" width="140"/><img src="media/demo_img/img2seq_rnn_2/moving_paint1.gif" alt="drawing" width="140"/><img src="media/demo_img/statue.png" alt="drawing" width="200"/><img src="media/demo_img/img2seq_rnn_1/moving_statue2.gif" alt="drawing" width="140"/><img src="media/demo_img/img2seq_rnn_2/moving_statue2.gif" alt="drawing" width="140"/>
</p>
 
## Citation

## References

#### Face Alignment
_A. Bulat and G. Tzimiropoulos, “How far are we from solving the 2d & 3d face alignment problem? (and a dataset of 230,000 3d facial landmarks),” in ICCV, 2017._
<!--  #### ACTOR
_M. Petrovich, M. J Black, and G. Varol, “Action-conditioned 3d human motion synthesis with transformer vae,” in ICCV, 2021._ -->
#### CONFER DB
_C. Georgakis, Y. Panagakis, S. Zafeiriou, and M. Pantic, “The conflict escalation resolution (confer) database,” Image and Vision Computing, vol. 65, 2017._
#### VoxCeleb2
_J. S. Chung, A. Nagrani, and A. Zisserman, “Voxceleb2: Deep speaker recognition,” in INTERSPEECH, 2018._
