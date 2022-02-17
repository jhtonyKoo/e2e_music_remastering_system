# End-to-end Music Remastering System

This repository includes source code and pre-trained models of the work *End-to-end Music Remastering System Using Self-supervised and Adversarial Training* by Junghyun Koo, Seungryeol Paik, and Kyogu Lee.

We provide inference code of the proposed system, which targets to alter the mastering style of a song to desired reference track.



[![arXiv](https://img.shields.io/badge/arXiv-****.*****-b31b1b.svg)](https://arxiv.org/)
[![Demo Page](https://img.shields.io/badge/Demo%20Page-****.*****-green.svg)](https://dg22302.github.io/MusicRemasteringSystem/)



## Pre-trained Models
| Model | Number of Epochs Trained | Details |
|-------------|-------------|-------------|
[Music Effects Encoder](https://drive.google.com/file/d/1vUFWEGDy3zS590puGTA2GRfXs-Aa-LFs/view?usp=sharing) | 1000 | Trained with [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) Dataset
[Mastering Cloner](https://drive.google.com/file/d/1kdPK2PO5mYEZ-7XlTWhP0p_oAynob0qM/view?usp=sharing) | 1000 | Trained with the above pre-trained Music Effects Encoder and Projection Discriminator



## Inference
To run the inference code, 
1. Download pre-trained models above and place them under the folder named 'model_checkpoints' (default)
2. Prepare input and reference tracks under the folder named 'inference_samples' (default).  
Target files should be organized as follow:
```
    "path_to_data_directory"/"song_name_#1"/input.wav
    "path_to_data_directory"/"song_name_#1"/reference.wav
    ...
    "path_to_data_directory"/"song_name_#n"/input.wav
    "path_to_data_directory"/"song_name_#n"/reference.wav
```
3. Run 'inference.py'
```
python inference.py \
    --ckpt_dir "path_to_checkpoint_directory" \
    --data_dir_test "path_to_directory_containing_inference_samples"
```
4. Outputs will be stored under the folder 'inference_samples' (default)

*Note: The system accepts WAV files of stereo-channeled, 44.1kHZ, and 16-bit rate. Target files shold be named "input.wav" and "reference.wav".*



## Configurations of each sub-networks

<div align="center">
  <img width="50%" alt="config_table" src="https://github.com/jhtonyKoo/e2e_music_remastering_system/blob/main/img/configuration_table.png">
</div>
<div align="center">
</div>


A detailed configuration of each sub-networks can also be found at
```
Self_Supervised_Music_Remastering_System/configs.yaml
```

<!-- ## Cite
Please consider citing the following work upon the usage of this repository.
```

``` -->
