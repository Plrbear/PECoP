import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.ucf101 import PECoP_SSL
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
from tensorboardX import SummaryWriter
from torchvideotransforms import video_transforms, volume_transforms
from models.i3d_adapter import I3D
from ptflops import get_model_complexity_info
from models.i3d_adapter import Unit3Dpy



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='gpu id')
    parser.add_argument('--height', type=int, default=256, help='resize height')
    parser.add_argument('--pretrained_i3d_weight', type=str, default='./model_rgb.pth', help='Specify the path to the I3D weights pretrained on K400 on your machine.')
    parser.add_argument('--width', type=int, default=256, help='resize width')
    parser.add_argument('--clip_len', type=int, default=32, help='64, input clip length')
    parser.add_argument('--crop_sz', type=int, default=224, help='crop size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='32, batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--epoch', type=int, default=8, help='total epoch')
    parser.add_argument('--max_sr', type=int, default=5, help='largest sampling rate- max playback rate [e.g. 5 means 1,2,3,4,5]')
    parser.add_argument('--max_segment', type=int, default=4, help='largest segments-')
    parser.add_argument('--fr', type=int, default=3, help='default sampling rate of videos- for gait we consider the default sampling rate 3 because of long video lenghth- please consider one for other tasks, e.g. leg agility, diving.')
    parser.add_argument('--max_save', type=int, default=100, help='max save epoch num')
    parser.add_argument('--pf', type=int, default=20, help='print frequency')
    parser.add_argument('--dataset', type=str, default='PD4T', help='dataset name')
    parser.add_argument('--model', type=str, default='i3d', help='s3d/r21d/r3d/c3d, pretrain model')
    parser.add_argument('--data_list', type=str, default='/home/amir/AQA/Datasets/PD/30_subjects_frames/frames/gait/label/train.list', help='labels')
    parser.add_argument('--rgb_prefix', type=str, default='/home/amir/AQA/Datasets/PD/30_subjects_frames/frames/', help='frames')

    args = parser.parse_args()

    return args



def get_mlp(inp_dim, out_dim):
    mlp = nn.Sequential(
        nn.BatchNorm1d(inp_dim),
        nn.ReLU(inplace=True),
        nn.Linear(inp_dim, out_dim),
)
    return mlp

def get_mlp_s(inp_dim, out_dim):
    mlp = nn.Sequential(
        nn.BatchNorm1d(inp_dim),
        nn.ReLU(inplace=True),
        nn.Linear(inp_dim, out_dim),
)
    return mlp    




class VSPP(nn.Module):
    def __init__(self, num_classes_p=5,num_classes_s=4):
        super(VSPP, self).__init__()



        self.num_classes_s=num_classes_s
        self.num_classes_p=num_classes_p


        self.dropout = nn.Dropout(p=0.5)  
        self.logits_p = Unit3Dpy(  #for the segment prediction head
            in_channels=1024,
            out_channels=num_classes_p,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)


        self.logits_s = Unit3Dpy( #for the video playback prediction head
            in_channels=1024,
            out_channels=self.num_classes_s,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)


#loading the model and loading the pretraining weights - I3D with 3D-Adapters

        self.model = I3D(num_classes= 400, dropout_prob=0.5)


        cp = torch.load(args.pretrained_i3d_weight)


        self.model.load_state_dict(cp,strict=False)

##############################################################################

        for param in self.model.parameters():  #Here we freez all the model's layers
            param.requires_grad = False

        # for a,b in self.model.state_dict().items():
        #     print(b)
        # exit()    

#########################here we unfreez the adapter layers###################################### 

        for param in self.model.mixed_3b.tuning_module.parameters():
            param.requires_grad = True
        for param in self.model.mixed_3c.tuning_module.parameters():
            param.requires_grad = True            
        for param in self.model.mixed_4b.tuning_module.parameters():
            param.requires_grad = True 
        for param in self.model.mixed_4c.tuning_module.parameters():
            param.requires_grad = True
        for param in self.model.mixed_4d.tuning_module.parameters():
            param.requires_grad = True
        for param in self.model.mixed_4e.tuning_module.parameters():
            param.requires_grad = True            
        for param in self.model.mixed_4f.tuning_module.parameters():
            param.requires_grad = True 
        for param in self.model.mixed_5c.tuning_module.parameters():
            param.requires_grad = True 
        for param in self.model.mixed_5b.tuning_module.parameters():
            param.requires_grad = True        

######################################################################################## 
        
    def forward(self, x): ##########adding classification heads##############
        x = self.model(x)
        x_p = self.logits_p(self.dropout(x))
        x_s = self.logits_s(self.dropout(x))

        l_p = x_p.squeeze(3).squeeze(3)
        l_s = x_s.squeeze(3).squeeze(3)
        l_p = torch.mean(l_p, 2)
        l_s = torch.mean(l_s, 2)

        return l_p,l_s


#################################


def train(args):
    torch.backends.cudnn.benchmark = True

    exp_name = '{}_sr_{}_{}_lr_{}_len_{}_sz_{}'.format(args.dataset, args.max_sr, args.model, args.lr, args.clip_len, args.crop_sz)


    print(exp_name)

    pretrain_cks_path = os.path.join('pretrain_cks', exp_name)
    log_path = os.path.join('visual_logs', exp_name)

    if not os.path.exists(pretrain_cks_path):
        os.makedirs(pretrain_cks_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)


    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # video_transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    color_jitter = transforms.RandomApply([color_jitter], p=0.8)

    train_dataset = PECoP_SSL(args.data_list, args.rgb_prefix, clip_len=args.clip_len, max_sr=args.max_sr, max_segment=args.max_segment,
                                   transforms_=train_trans, color_jitter_=color_jitter, fr=args.fr)

    print("len of training data:", len(train_dataset))
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)

  
    if args.model == 'i3d':

 #####################loading the whole model with the heads#########


        model = VSPP(num_classes_p=args.max_sr,num_classes_s=args.max_segment)



    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

######################################################################################


    criterion = nn.CrossEntropyLoss()
 


#########################################################################################


    model.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_dir=log_path)
    iterations = 1





    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        dampening=0,
        weight_decay=1e-4,
        nesterov=False,
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=22,
        eta_min=args.lr / 1000
    )




    model.train()

    for epoch in range(args.epoch):
        total_loss = 0.0
        correct = 0
        it=0

        for i, sample in enumerate(dataloader):
            rgb_clip, labels = sample
            rgb_clip = rgb_clip.to(device, dtype=torch.float)
            label_speed = labels[:,0].to(device)
            label_segment = labels[:,1].to(device)


            optimizer.zero_grad()
            out1, out2 = model(rgb_clip)
     
            loss1 = criterion(out1, label_speed)
            loss2 = criterion(out2, label_segment)
            loss = loss1 + loss2

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            probs_segment = nn.Softmax(dim=1)(out2)
            preds_segment = torch.max(probs_segment, 1)[1]
            accuracy_seg = torch.sum(preds_segment == label_segment.data).detach().cpu().numpy().astype(np.float)

            probs_speed = nn.Softmax(dim=1)(out1)
            preds_speed = torch.max(probs_speed, 1)[1]
            accuracy_speed = torch.sum(preds_speed == label_speed.data).detach().cpu().numpy().astype(np.float)
            accuracy = ((accuracy_speed + accuracy_seg)/2) / args.bs
            correct += ((accuracy_speed + accuracy_seg)/2) / args.bs

            iterations += 1
            it += 1

            if i % args.pf == 0:
                writer.add_scalar('data/train_loss', loss, iterations)
                writer.add_scalar('data/Acc', accuracy, iterations)

                print("[Epoch{}/{}] Loss: {} Acc: {}  ".format(
                    epoch + 1, i, loss, accuracy))

        
        avg_loss = total_loss / it
        avg_acc = correct / it
        print('[pre-training] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))    

        scheduler.step()
        model_saver(model, optimizer, epoch, args.max_save, pretrain_cks_path)

    writer.close()


def model_saver(net, optimizer, epoch, max_to_keep, model_save_path):
    tmp_dir = os.listdir(model_save_path)
    # print(tmp_dir)
    tmp_dir.sort()
    if len(tmp_dir) >= max_to_keep:
        os.remove(os.path.join(model_save_path, tmp_dir[0]))

    torch.save(net.state_dict(), os.path.join(model_save_path, 'gait_adapter' + '{:02}'.format(epoch + 1) + '.pth.tar'))


if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train(args)
