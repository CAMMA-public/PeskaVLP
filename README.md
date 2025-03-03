<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="https://github.com/CAMMA-public/SelfSupSurg/blob/main/static/camma_logo_tr.png" width="30%">
</a>
</div>

# **Procedure-Aware Surgical Video-language Pretraining with Hierarchical Knowledge Augmentation**
_Kun Yuan, Vinkle Srivastav, Nassir Navab, Nicolas Padoy_,  **NeurIPS 2024** 

[![arXiv](https://img.shields.io/badge/arxiv-2307.15220-red)](https://arxiv.org/abs/2410.00263) [OpenReview](https://openreview.net/forum?id=zuwpeRkJNH&noteId=IgTHiK4DS9)



This repo contains an open source PyTorch distributed training code for: 
- Under Review Medical Image Analysis paper: [Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures](https://arxiv.org/abs/2307.15220) [1]
- MICCAI'24 paper: [HecVL: Hierarchical Video-Language Pretraining for Zero-shot Surgical Phase Recognition](https://arxiv.org/abs/2405.10075) [2]
- NeurIPS'24 Spotlight paper: [Procedure-Aware Surgical Video-language Pretraining with Hierarchical Knowledge Augmentation](https://arxiv.org/abs/2410.00263) [3]

This repository provides an implementation using PyTorch and ffmpeg with a reasonable number of GPUs. The training code was runned on the French public AI cluster Jean-Zay, which is managed by the SLURM system, and also supports single GPU training. We extend our gratitude to the community for their prior contributions, which have been instrumental in developing this project: [MIL-NCE](https://github.com/antoine77340/MIL-NCE_HowTo100M) and [DeCLIP](https://github.com/Sense-GVT/DeCLIP) 

## 
If you only plan to reuse the pretrained surgical vision-language model from [1,2,3], please visit the following [repo](https://github.com/CAMMA-public/SurgVLP/) If you use this code, we would appreciate if you could cite our papers:
```bibtex
@article{yuan2023learning,
  title={Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures},
  author={Yuan, Kun and Srivastav, Vinkle and Yu, Tong and Lavanchy, Joel and Mascagni, Pietro and Navab, Nassir and Padoy, Nicolas},
  year={2023},
  eprint={2307.15220},
  archivePrefix={arXiv}
}

@inproceedings{yuan2024hecvl,
  title={HecVL: hierarchical video-language pretraining for zero-shot surgical phase recognition},
  author={Yuan, Kun and Srivastav, Vinkle and Navab, Nassir and Padoy, Nicolas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={306--316},
  year={2024},
  organization={Springer}
}

@article{yuan2025procedure,
  title={Procedure-aware surgical video-language pretraining with hierarchical knowledge augmentation},
  author={Yuan, Kun and Navab, Nassir and Padoy, Nicolas and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={122952--122983},
  year={2025}
}
```

## Requirements
- Python 3
- PyTorch (>= 2.0)
- [python-ffmpeg](https://github.com/kkroening/ffmpeg-python) with ffmpeg 
- mmengine==0.7.0
- torchvision
- numpy<2
- torchmetrics
- pycm
- fvcore
- transformers==4.30.2
- scikit-learn
- tqdm


## Step1: Prepare Data
To create your own dataset, follow the structure of the 'data' folder. Each sub-folder is organized as follows:
- Training:
  - videos: store the video in .mp4 format
  - whisper_narration_texts: store the narration texts from OpenAI Whisper for each video
  - aws_narration_texts: store narration texts from AWS Medical Transcribe for each video
  - keystep_texts: store keystep texts for each video
  - abstract_texts: store abstract texts for each video
  - splits: csv files that sprcify which videos are used for training or validating for clip-/keystep-/video-level of pretraining
- Evaluation:
  - prompts: class prompts for downstream datasets
  - downstream_datasets:
    - CholecT50-challenge-train: data downloaded from [CholecT50](https://github.com/CAMMA-public/cholect50/blob/master/docs/README-Downloads.md)
    - cholec80:
      - csvs: annotation files for each video
      - frames: frames named as {video_id}_{frame_id}.png (download and extract in 1 FPS from [Cholec80](https://github.com/CAMMA-public/TF-Cholec80))
    - autolaparo
      - csvs: annotation files for each video (download from [Autolaparo](https://autolaparo.github.io/))
      - frames: frames named as {video_id}_{frame_id}.png

## Step2: Prepare Config File
Before runnig the code, you need to create a folder and init a 'config.py' file. Examples are given in the 'configs' folder.
- ./configs/surgvlp/config.py : the configuration to the [1]
- ./configs/surgvlp/hecvl.py : the configuration to the [2]
- ./configs/surgvlp/peskavlp.py : the configuration to the [3]

## Step3: Training Your Own Model
The following command line trains the surgical vision-language model on a single node for [1,2,3]. It uses all of its GPU and save the model checkpoints in the directory configs/'surgvlp/hecvl/peskavlp', the log is created and written in the *log.txt* directory. A 'tensorboard' directory is created and logs are recorded inside. '--resume' allows you to automatically find the last checkpoint and resume from that when your training is interrupted, you can always set it True.

SurgVLP [1]:
```python
python train.py --resume --work_dir=./configs/surgvlp/
```

HecVL [2]:
```python
python train_hierarchy_ssl.py --resume --work_dir=./configs/hecvl/
```

PeskaVLP [3]:
```python
python train_hierarchy_ssl.py --resume --work_dir=./configs/peskavlp/
```
## License
The code and the models are available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.

By downloading and using this repo, you agree to these terms and conditions.

## Acknowledgement
This work has received funding from the European Union
(ERC, CompSURG, 101088553). Views and opinions expressed are however those of the authors only and do not
necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor
the granting authority can be held responsible for them. This work was also partially supported by French state funds
managed by the ANR under Grants ANR-20-CHIA-0029-01 and ANR-10-IAHU-02.

