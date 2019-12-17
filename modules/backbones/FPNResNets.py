'''
Function:
	Feature Pyramid Network of ResNets
Author:
	Charles
'''
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


'''resnet from torchvision==0.2.2'''
def ResNets(resnet_type, pretrained=False):
	if resnet_type == 'resnet18':
		model = torchvision.models.resnet18(pretrained=pretrained)
	elif resnet_type == 'resnet34':
		model = torchvision.models.resnet34(pretrained=pretrained)
	elif resnet_type == 'resnet50':
		model = torchvision.models.resnet50(pretrained=pretrained)
	elif resnet_type == 'resnet101':
		model = torchvision.models.resnet101(pretrained=pretrained)
	elif resnet_type == 'resnet152':
		model = torchvision.models.resnet152(pretrained=pretrained)
	else:
		raise ValueError('Unsupport resnet_type <%s>...' % resnet_type)
	return model


'''FPN by using ResNets'''
class FPNResNets(nn.Module):
	def __init__(self, mode, cfg, logger_handle, **kwargs):
		super(FPNResNets, self).__init__()
		self.logger_handle = logger_handle
		self.pretrained_model_path = cfg.PRETRAINED_MODEL_PATH
		self.backbone = ResNets(resnet_type=cfg.BACKBONE_TYPE, pretrained=False)
		if cfg.WEIGHTS_NEED_INITIALIZE and mode == 'TRAIN':
			self.initializeBackbone()
		self.backbone.avgpool = None
		self.backbone.fc = None
		# parse backbone
		self.base_layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
		self.base_layer1 = nn.Sequential(self.backbone.layer1)
		self.base_layer2 = nn.Sequential(self.backbone.layer2)
		self.base_layer3 = nn.Sequential(self.backbone.layer3)
		self.base_layer4 = nn.Sequential(self.backbone.layer4)
		# add lateral layers
		self.lateral_layer0 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.lateral_layer1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.lateral_layer2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.lateral_layer3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
		# add smooth layers
		self.smooth_layer1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.smooth_layer2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.smooth_layer3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
		# add downsample layer
		self.downsample_layer = nn.MaxPool2d(kernel_size=1, stride=2)
	'''forward'''
	def forward(self, x):
		# bottom-up
		c1 = self.base_layer0(x)
		c2 = self.base_layer1(c1)
		c3 = self.base_layer2(c2)
		c4 = self.base_layer3(c3)
		c5 = self.base_layer4(c4)
		# top-down
		p5 = self.lateral_layer0(c5)
		p4 = self.upsampleAdd(p5, self.lateral_layer1(c4))
		p4 = self.smooth_layer1(p4)
		p3 = self.upsampleAdd(p4, self.lateral_layer2(c3))
		p3 = self.smooth_layer2(p3)
		p2 = self.upsampleAdd(p3, self.lateral_layer3(c2))
		p2 = self.smooth_layer3(p2)
		p6 = self.downsample_layer(p5)
		# return all feature pyramid levels
		return [p2, p3, p4, p5, p6]
	'''upsample and add'''
	def upsampleAdd(self, p, c):
		_, _, H, W = c.size()
		return F.interpolate(p, size=(H, W), mode='bilinear') + c
	'''initialize model'''
	def initializeBackbone(self):
		if self.pretrained_model_path:
			self.backbone.load_state_dict({k:v for k,v in torch.load(self.pretrained_model_path).items() if k in self.backbone.state_dict()})
			self.logger_handle.info('Loading pretrained weights from %s for backbone network...' % self.pretrained_model_path)
		else:
			self.backbone = ResNets(resnet_type=self.backbone_type, pretrained=True)