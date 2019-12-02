'''
Function:
	Feature Pyramid Network of ResNets
Author:
	Charles
'''
import torch
import torch.nn as nn
from modules.backbones import *


'''ResNetsFPN'''
class ResNetsFPN(nn.Module):
	def __init__(self, mode, cfg, logger_handle, **kwargs):
		super(ResNetsFPN, self).__init__()
		self.logger_handle = logger_handle
		self.pretrained_model_path = cfg.PRETRAINED_MODEL_PATH
		self.backbone = ResNets(resnet_type=cfg.BACKBONE_TYPE, pretrained=False)
		if cfg.WEIGHTS_NEED_INITIALIZE and mode == 'TRAIN':
			self.initializeBackbone()
		self.backbone.avgpool = None
		self.backbone.fc = None


	'''initialize model'''
	def initializeBackbone(self):
		if self.pretrained_model_path:
			self.backbone.load_state_dict({k:v for k,v in torch.load(self.pretrained_model_path).items() if k in self.backbone.state_dict()})
			self.logger_handle.info('Loading pretrained weights from %s for backbone network...' % self.pretrained_model_path)
		else:
			self.backbone_type = ResNets(resnet_type=self.backbone_type, pretrained=True)