# PECoP
This repository contains the PD4T dataset and PyTorch implementation for **PECoP: Parameter Efficient Continual Pretraining for Action Quality Assessment** (WACV 2024, Oral) [[arXiv](https://arxiv.org/pdf/2311.07603.pdf)]



## PD4T Dataset 


![PD4T](https://github.com/Plrbear/PECoP/assets/31938815/80ba7e89-72be-4353-b933-1659773c9fdb)


PD4T is a Parkinson’s disease dataset for human action quality assessment. videos were recorded using a single RGB camera from 30 PD patients performed four different PD tasks including gait (426 videos), hand movement (848 videos), finger tapping (806 videos), and leg agility (851 videos) in clinical settings.
The trained rater assigned a score for each task, based on the protocols of UPDRS, varying between 0 and 4 depending on the level of severity. 
### Download
To access the PD4T dataset, please complete and sign the [PD4T request form](datasets/PD4T_Request_Form.docx) and forward it to a.dadashzadeh@bristol.ac.uk. By submitting your application, you acknowledge and confirm that you have read and understood the relevant notice. Upon receiving your request, we will promptly respond with the necessary link and guidelines. **Please note that we only review requests that are submitted by faculty members.**
 
## Pretraining via PECoP 
PECoP vs Baseline: Enhancing AQA with domain-specific pretraining.



<img src="https://github.com/Plrbear/PECoP/assets/31938815/83b9a5f0-04cc-4c12-90da-39cbf438ec87" width="400">



### Requirements
- pytroch >= 1.3.0
- tensorboardX
- cv2
- scipy



### Pretraining
The Kinetics pretrained I3D downloaded from the reposity [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth).

To train I3D model using PECoP, update the paths for the Kinetics pretrained I3D and your data directory accordingly. Then, execute the following command:

python train.py --bs 16 --lr 0.001 --height 256 --width 256 --crop_sz 224 --clip_len 32



### Fine-tuning and Evaluation
The pretrained I3D model with PECoP serves as the backbone network for our baselines — [USDL](https://github.com/nzl-thu/MUSDL) , [MUSDL](https://github.com/nzl-thu/MUSDL), [CoRe](https://github.com/yuxumin/CoRe), and [TSA](https://github.com/xujinglin/FineDiving) — and their performance is evaluated using the Spearman Rank Correlation metric.

### Results
Spearman Rank Correlation results on the PD4T dataset with two AQA baselines, CoRe and USDL.

<img src="https://github.com/Plrbear/PECoP/assets/31938815/f5b017b7-6f9b-45a0-9237-1a2b95bb424d" width="600">


## Citation
If you find this work useful or use our code, please consider citing:

```
@article{dadashzadeh2023pecop,
  title={PECoP: Parameter Efficient Continual Pretraining for Action Quality Assessment},
  author={Dadashzadeh, Amirhossein and Duan, Shuchao and Whone, Alan and Mirmehdi, Majid},
  journal={arXiv preprint arXiv:2311.07603},
  year={2023}
}
```



## Contact
 For any question, please file an issue or send an email to: a.dadashzadeh@bristol.ac.uk


## Acknowlegement
The authors would like to gratefully acknowledge the
contribution of the Parkinson’s study participants. The 1st
author is funded by a kind donation made to Southmead
Hospital Charity, Bristol and by a donation from Caroline
Belcher via the University of Bristol’s Development and
Alumni Relations. The clinical trial from which the video
data of the people with Parkinson’s was sourced was funded
by Parkinson’s UK (Grant J-1102), with support from Cure
Parkinson’s.


Part of our codes are adapted from [VideoPace](https://github.com/laura-wang/video-pace), we thank the authors for their contributions.

