# PECoP: Parameter Efficient Continual Pretraining for Action Quality Assessment

# PD4T Dataset 


![PD4T](https://github.com/Plrbear/PECoP/assets/31938815/80ba7e89-72be-4353-b933-1659773c9fdb)


PD4T is a Parkinson’s disease dataset for human action quality assessment. videos were recorded using a single RGB camera from 30 PD patients performed four different PD tasks including gait (426 videos), hand movement (848 videos), finger tapping (806 videos), and leg agility (851 videos) in clinical settings.
The trained rater assigned a score for each task, based on the protocols of UPDRS, varying between 0 and 4 depending on the level of severity. 
# Download
To access the PD4T dataset, please complete and sign the [PD4T request form](datasets/PD4T_Request_Form.docx) and forward it to a.dadashzadeh@bristol.ac.uk. By submitting your application, you acknowledge and confirm that you have read and understood the relevant notice. Upon receiving your request, we will promptly respond with the necessary link and guidelines.

# Requirements
- pytroch >= 1.3.0
- tensorboardX
- cv2
- scipy



## Pretrain
will be updated...

python train.py --bs 16 --lr 0.001 --height 256 --width 256 --crop_sz 224 --clip_len 32



## Evaluation
Will be updaated...

# Citation
Will be updated....

# Acknowlegement
Will be updated...


