# PECoP: Parameter Efficient Continual Pretraining for Action Quality Assessment

# Dataset


### The PD4T Dataset Summary, Categorized by Severity Scores

### The PD4T Dataset Summary, Categorized by Severity Scores

For each of the four motor tasks, the table lists the total number of videos (#video), the minimum (#min) and maximum (#max) number of frames for the respective task.

|                      |        | Normal (0) | Slight (1) | Mild (2) | Moderate (3) | Severe (4) |
|----------------------|--------|------------|------------|----------|--------------|------------|
| <div align="center">**Gait**</div> | #video | 196        | 158        | 64       | 8            | 0          |
|                      | #min   | 325        | 580        | 421      | 664          | -          |
|                      | #max   | 980        | 1866       | 13428    | 10688        | -          |
|----------------------|--------|------------|------------|----------|--------------|------------|
| <div align="center">**Finger Tapping**</div>   | #video | 152        | 465        | 164      | 23           | 2          |
|                      | #min   | 129        | 129        | 129      | 162          | 159        |
|                      | #max   | 450        | 724        | 853      | 398          | 460        |
|----------------------|--------|------------|------------|----------|--------------|------------|
| <div align="center">**Hand Movements**</div>   | #video | 234        | 407        | 179      | 23           | 5          |
|                      | #min   | 131        | 136        | 150      | 197          | 220        |
|                      | #max   | 334        | 571        | 717      | 648          | 648        |
|----------------------|--------|------------|------------|----------|--------------|------------|
| <div align="center">**Leg Agility**</div>      | #video | 407        | 376        | 54       | 11           | 3          |
|                      | #min   | 129        | 135        | 155      | 273          | 345        |
|                      | #max   | 513        | 427        | 686      | 504          | 435        |


# Requirements
- pytroch >= 1.3.0
- tensorboardX
- cv2
- scipy


## Data preparation
Will be updated....


## Pretrain
will be updated...

python train.py --bs 16 --lr 0.001 --height 256 --width 256 --crop_sz 224 --clip_len 32



## Evaluation
Will be updaated...

# Citation
Will be updated....

# Acknowlegement
Will be updated...


