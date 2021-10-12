import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, load_checkpoint

from mmdet.models.builder import SHARED_HEADS, build_loss
from mmdet.utils import get_root_logger

@SHARED_HEADS.register_module()
class AET_head(nn.Module):
    '''The decoder of AET branches, input the feat of original images and 
    feat of transformed images, passed by global pool and return the transformed
    results'''
    def __init__(self,
                 indim= 2048, 
                 num_classes=4,
                 ):
        super(AET_head, self).__init__()
        self.indim = indim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(indim, int(indim/2))
        self.fc2 = nn.Linear(int(indim/2), num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def global_pool(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)
    
    def forward(self, feat1, feat2):
        feat1 = self.global_pool(feat1)
        feat2 = self.global_pool(feat2)
        device = feat1[0][0].device
        #print(device)
        x = torch.cat((feat1, feat2), dim=1)
        # print(x.shape)
        x = self.fc1.to(torch.device(device))(x)
        x = self.fc2.to(torch.device(device))(x)
        return x
    
    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in LinearBlock
        pass
    

        


