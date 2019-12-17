'''
Function:
	define the faster RCNN
Author:
	Charles
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.backbones import *
from modules.utils.utils import *
from modules.losses.smoothL1 import *
from modules.RPN import RegionProposalNet
from libs.roi_crop.modules.roi_crop import RoICrop
from libs.roi_align.modules.roi_align import RoIAlignAvg
from libs.roi_pooling.modules.roi_pool import RoIPooling


'''build proposal target layer'''
class buildProposalTargetLayer(nn.Module):
	def __init__(self, mode, cfg, **kwargs):
		super(buildProposalTargetLayer, self).__init__()
		self.mode = mode
		self.cfg = cfg
		if mode == 'TRAIN':
			self.bbox_normalize_means = torch.FloatTensor(cfg.TRAIN_BBOX_NORMALIZE_MEANS)
			self.bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN_BBOX_NORMALIZE_STDS)
			self.roi_batchsize = cfg.TRAIN_ROI_BATCHSIZE
			self.roi_fg_fraction = cfg.TRAIN_ROI_FG_FRACTION
			self.roi_fg_thresh = cfg.TRAIN_ROI_FG_THRESH
			self.roi_bg_thresh_hi = cfg.TRAIN_ROI_BG_THRESH_HI
			self.roi_bg_thresh_lo = cfg.TRAIN_ROI_BG_THRESH_LO
		elif mode == 'TEST':
			self.bbox_normalize_means = torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_MEANS)
			self.bbox_normalize_stds = torch.FloatTensor(cfg.TEST_BBOX_NORMALIZE_STDS)
			self.roi_batchsize = cfg.TEST_ROI_BATCHSIZE
			self.roi_fg_fraction = cfg.TEST_ROI_FG_FRACTION
			self.roi_fg_thresh = cfg.TEST_ROI_FG_THRESH
			self.roi_bg_thresh_hi = cfg.TEST_ROI_BG_THRESH_HI
			self.roi_bg_thresh_lo = cfg.TEST_ROI_BG_THRESH_LO
		else:
			raise ValueError('Unkown mode <%s> in buildProposalTargetLayer...' % mode)
		self.bbox_inside_weights = torch.FloatTensor([1., 1., 1., 1.])
	def forward(self, all_rois, gt_boxes, num_gt_boxes):
		self.bbox_normalize_means = self.bbox_normalize_means.type_as(gt_boxes)
		self.bbox_normalize_stds = self.bbox_normalize_stds.type_as(gt_boxes)
		self.bbox_inside_weights = self.bbox_inside_weights.type_as(gt_boxes)
		gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
		gt_boxes_append[..., 1:5] = gt_boxes[..., :4]
		all_rois = torch.cat([all_rois, gt_boxes_append], 1)
		num_rois_per_image = self.roi_batchsize
		num_rois_fg_per_image = int(np.round(self.roi_fg_fraction * num_rois_per_image))
		rois, labels, bbox_targets, bbox_inside_weights = self.sampleRoIs(all_rois, gt_boxes, num_rois_fg_per_image, num_rois_per_image)
		bbox_outside_weights = (bbox_inside_weights > 0).float()
		return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
	def sampleRoIs(self, all_rois, gt_boxes, num_rois_fg_per_image, num_rois_per_image):
		overlaps = BBoxFunctions.calcIoUs(all_rois, gt_boxes)
		max_overlaps, gt_assignment = torch.max(overlaps, 2)
		batch_size = overlaps.size(0)
		offsets = torch.arange(0, batch_size) * gt_boxes.size(1)
		offsets = offsets.view(-1, 1).type_as(gt_assignment) + gt_assignment
		labels = gt_boxes[..., 4].contiguous().view(-1)[(offsets.view(-1),)].view(batch_size, -1)
		labels_batch = labels.new(batch_size, num_rois_per_image).zero_()
		rois_batch = all_rois.new(batch_size, num_rois_per_image, 5).zero_()
		gt_rois_batch = all_rois.new(batch_size, num_rois_per_image, 5).zero_()
		for i in range(batch_size):
			fg_idxs = torch.nonzero(max_overlaps[i] >= self.roi_fg_thresh).view(-1)
			num_rois_fg = fg_idxs.numel()
			bg_idxs = torch.nonzero((max_overlaps[i] < self.roi_bg_thresh_hi) & (max_overlaps[i] >= self.roi_bg_thresh_lo)).view(-1)
			num_rois_bg = bg_idxs.numel()
			if num_rois_fg > 0 and num_rois_bg > 0:
				num_rois_fg_this_image = min(num_rois_fg_per_image, num_rois_fg)
				rand_num = torch.from_numpy(np.random.permutation(num_rois_fg)).type_as(gt_boxes).long()
				fg_idxs = fg_idxs[rand_num[:num_rois_fg_this_image]]
				num_rois_bg_this_image = num_rois_per_image - num_rois_fg_this_image
				rand_num = np.floor(np.random.rand(num_rois_bg_this_image) * num_rois_bg)
				rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
				bg_idxs = bg_idxs[rand_num]
			elif num_rois_fg > 0 and num_rois_bg == 0:
				rand_num = np.floor(np.random.rand(num_rois_per_image) * num_rois_fg)
				rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
				fg_idxs = fg_idxs[rand_num]
				num_rois_fg_this_image = num_rois_per_image
				num_rois_bg_this_image = 0
			elif num_rois_fg == 0 and num_rois_bg > 0:
				rand_num = np.floor(np.random.rand(num_rois_per_image) * num_rois_bg)
				rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
				bg_idxs = bg_idxs[rand_num]
				num_rois_fg_this_image = 0
				num_rois_bg_this_image = num_rois_per_image
			else:
				raise ValueError('num_rois_fg = 0 and num_rois_bg = 0, your program should be wrong somewhere...')
			keep_idxs = torch.cat([fg_idxs, bg_idxs], 0)
			labels_batch[i].copy_(labels[i][keep_idxs])
			if num_rois_fg_this_image < num_rois_per_image:
				labels_batch[i][num_rois_fg_this_image:] = 0
			rois_batch[i] = all_rois[i][keep_idxs]
			rois_batch[i, :, 0] = i
			gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_idxs]]
		bbox_targets = self.computeTargets(rois_batch[..., 1:5], gt_rois_batch[..., :4])
		bbox_targets, bbox_inside_weights = self.getBBoxRegressionTargets(bbox_targets, labels_batch)
		return rois_batch, labels_batch, bbox_targets, bbox_inside_weights
	def computeTargets(self, ex_rois, gt_rois):
		assert ex_rois.size(1) == gt_rois.size(1)
		assert ex_rois.size(2) == 4
		assert gt_rois.size(2) == 4
		batch_size = ex_rois.size(0)
		num_rois_per_image = ex_rois.size(1)
		targets = BBoxFunctions.encodeBboxes(ex_rois, gt_rois)
		targets = ((targets - self.bbox_normalize_means.expand_as(targets)) / self.bbox_normalize_stds.expand_as(targets))
		return targets
	def getBBoxRegressionTargets(self, bbox_targets, labels_batch):
		batch_size = labels_batch.size(0)
		num_rois_per_image = labels_batch.size(1)
		bbox_targets_new = bbox_targets.new(batch_size, num_rois_per_image, 4).zero_()
		bbox_inside_weights = bbox_targets.new(bbox_targets_new.size()).zero_()
		for b in range(batch_size):
			if labels_batch[b].sum() == 0:
				continue
			idxs = torch.nonzero(labels_batch[b] > 0).view(-1)
			for i in range(idxs.numel()):
				idx = idxs[i]
				bbox_targets_new[b, idx, :] = bbox_targets[b, idx, :]
				bbox_inside_weights[b, idx, :] = self.bbox_inside_weights
		return bbox_targets_new, bbox_inside_weights
	def backward(self, *args):
		pass


'''base model for faster rcnn'''
class fasterRCNNFPNBase(nn.Module):
	def __init__(self, num_classes, is_class_agnostic, rpn_feature_strides, rcnn_feature_strides, mode, cfg, **kwargs):
		super(fasterRCNNFPNBase, self).__init__()
		self.num_classes = num_classes
		self.is_class_agnostic = is_class_agnostic
		self.rpn_feature_strides = rpn_feature_strides
		self.rcnn_feature_strides = rcnn_feature_strides
		self.mode = mode
		self.cfg = cfg
		if self.mode == 'TRAIN':
			self.pooling_method = cfg.TRAIN_POOLING_METHOD
			self.pooling_size = cfg.TRAIN_POOLING_SIZE
		elif self.mode == 'TEST':
			self.pooling_method = cfg.TEST_POOLING_METHOD
			self.pooling_size = cfg.TEST_POOLING_SIZE
		else:
			raise ValueError('Unkown mode <%s> in fasterRCNNFPNBase...' % mode)
		# base model
		self.base_model = None
		# RPN
		self.rpn_net = None
		self.roi_crop = None
		self.roi_align = None
		self.roi_pooling = None
		self.build_proposal_target_layer = None
		# top model
		self.top_model = None
		# final results
		self.fc_cls = None
		self.fc_loc = None
	'''forward'''
	def forward(self, x, gt_boxes, img_info, num_gt_boxes):
		batch_size = x.size(0)
		# extract features using backbone network
		p2, p3, p4, p5, p6 = self.base_model(x)
		rpn_features = [p2, p3, p4, p5, p6]
		rcnn_features = [p2, p3, p4, p5]
		# obtain rois
		rois, rpn_cls_loss, rpn_loc_loss = self.rpn_net(rpn_features, gt_boxes, img_info, num_gt_boxes)
		# if train
		if self.mode == 'TRAIN' and gt_boxes is not None:
			rois, rois_labels, rois_bbox_targets, rois_bbox_inside_weights, rois_bbox_outside_weights = self.build_proposal_target_layer(rois, gt_boxes, num_gt_boxes)
			rois_labels = rois_labels.view(-1).long()
			rois_bbox_targets = rois_bbox_targets.view(-1, rois_bbox_targets.size(2))
			rois_bbox_inside_weights = rois_bbox_inside_weights.view(-1, rois_bbox_inside_weights.size(2))
			rois_bbox_outside_weights = rois_bbox_outside_weights.view(-1, rois_bbox_outside_weights.size(2))
		else:
			rois_labels = None
			rois_bbox_targets = None
			rois_bbox_inside_weights = None
			rois_bbox_outside_weights = None
		# roi pooling based on obtained rois
		rois = rois.view(-1, 5)
		rois_h = rois.data[:, 4] - rois.data[:, 2] + 1
		rois_w = rois.data[:, 3] - rois.data[:, 1] + 1
		roi_levels = torch.log2(torch.sqrt(rois_h * rois_w) / 224.0)
		roi_levels = torch.round(roi_levels + 4)
		roi_levels[roi_levels < 2] = 2
		roi_levels[roi_levels > 5] = 5
		if self.pooling_method == 'crop':
			pooled_features = []
			boxes_levels = []
			for i, level in enumerate(range(2, 6)):
				if (roi_levels == level).sum() < 1.:
					continue
				keep_idxs_level = (roi_levels == level).nonzero().squeeze().view(-1)
				boxes_levels.append(keep_idxs_level)
				grid_size = self.pooling_size * 2
				grid_xy = fasterRCNNFPNBase.affineGridGen(rois[keep_idxs_level].view(-1, 5), rcnn_features[i].size()[2:], grid_size, self.rcnn_feature_strides[i])
				grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3)
				pooled_features.append(F.max_pool2d(self.roi_crop(rcnn_features[i], grid_yx.detach()), 2, 2))
			pooled_features = torch.cat(pooled_features, 0)
			boxes_levels = torch.cat(boxes_levels, 0)
			pooled_features = pooled_features[torch.sort(boxes_levels)[-1]]
		elif self.pooling_method == 'align':
			pooled_features = []
			boxes_levels = []
			for i, level in enumerate(range(2, 6)):
				if (roi_levels == level).sum() < 1.:
					continue
				keep_idxs_level = (roi_levels == level).nonzero().squeeze().view(-1)
				boxes_levels.append(keep_idxs_level)
				pooled_features.append(self.roi_align(rcnn_features[i], rois[keep_idxs_level].view(-1, 5), 1./self.rcnn_feature_strides[i]))
			pooled_features = torch.cat(pooled_features, 0)
			boxes_levels = torch.cat(boxes_levels, 0)
			pooled_features = pooled_features[torch.sort(boxes_levels)[-1]]
		elif self.pooling_method == 'pool':
			pooled_features = []
			boxes_levels = []
			for i, level in enumerate(range(2, 6)):
				if (roi_levels == level).sum() < 1.:
					continue
				keep_idxs_level = (roi_levels == level).nonzero().squeeze().view(-1)
				boxes_levels.append(keep_idxs_level)
				pooled_features.append(self.roi_pooling(rcnn_features[i], rois[keep_idxs_level].view(-1, 5), 1./self.rcnn_feature_strides[i]))
			pooled_features = torch.cat(pooled_features, 0)
			boxes_levels = torch.cat(boxes_levels, 0)
			pooled_features = pooled_features[torch.sort(boxes_levels)[-1]]
		else:
			raise ValueError('Unkown pooling_method <%s> in fasterRCNNFPNBase...' % self.pooling_method)
		# feed into top model
		pooled_features = self.top_model(pooled_features)
		if len(pooled_features.size()) == 4:
			pooled_features = pooled_features.mean(3).mean(2)
		# predict location
		x_loc = self.fc_loc(pooled_features)
		if self.mode == 'TRAIN' and not self.is_class_agnostic:
			x_loc = x_loc.view(x_loc.size(0), -1, 4)
			x_loc = torch.gather(x_loc, 1, rois_labels.view(rois_labels.size(0), 1, 1).expand(rois_labels.size(0), 1, 4))
			x_loc = x_loc.squeeze(1)
		# predict classification
		x_cls = self.fc_cls(pooled_features)
		cls_probs = F.softmax(x_cls, 1)
		# calculate loss
		loss_cls = torch.Tensor([0]).type_as(x)
		loss_loc = torch.Tensor([0]).type_as(x)
		if self.mode == 'TRAIN':
			# --classification loss
			if self.cfg.RCNN_CLS_LOSS_SET['type'] == 'cross_entropy':
				loss_cls = F.cross_entropy(x_cls, rois_labels, size_average=self.cfg.RCNN_CLS_LOSS_SET['cross_entropy']['size_average'])
				loss_cls = loss_cls * self.cfg.RCNN_CLS_LOSS_SET['cross_entropy']['weight']
			else:
				raise ValueError('Unkown classification loss type <%s>...' % self.cfg.RCNN_CLS_LOSS_SET['type'])
			# --regression loss
			if self.cfg.RCNN_REG_LOSS_SET['type'] == 'smoothL1Loss':
				loss_loc = smoothL1Loss(x_loc, rois_bbox_targets, rois_bbox_inside_weights, rois_bbox_outside_weights)
				loss_loc = loss_loc * self.cfg.RCNN_REG_LOSS_SET['smoothL1Loss']['weight']
			elif self.cfg.RCNN_REG_LOSS_SET['type'] == 'betaSmoothL1Loss':
				loss_loc = betaSmoothL1Loss(x_loc[rois_bbox_inside_weights>0].view(-1, 4), rois_bbox_targets[rois_bbox_inside_weights>0].view(-1, 4), beta=self.cfg.RCNN_REG_LOSS_SET['betaSmoothL1Loss']['beta'], size_average=self.cfg.RCNN_REG_LOSS_SET['betaSmoothL1Loss']['size_average'])
				loss_loc = loss_loc * self.cfg.RCNN_REG_LOSS_SET['betaSmoothL1Loss']['weight']
			else:
				raise ValueError('Unkown regression loss type <%s>...' % self.cfg.RCNN_REG_LOSS_SET['type'])
		rois = rois.view(batch_size, -1, rois.size(1))
		cls_probs = cls_probs.view(batch_size, rois.size(1), -1)
		bbox_preds = x_loc.view(batch_size, rois.size(1), -1)
		return rois, cls_probs, bbox_preds, rpn_cls_loss, rpn_loc_loss, loss_cls, loss_loc
	'''initialize except for backbone network'''
	def initializeAddedModules(self):
		if self.cfg.USE_CAFFE_PRETRAINED_MODEL and self.cfg.RCNN_REG_LOSS_SET['type'] == 'smoothL1Loss':
			fasterRCNNFPNBase.initWeights(self.base_model.lateral_layer0, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.base_model.lateral_layer1, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.base_model.lateral_layer2, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.base_model.lateral_layer3, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.base_model.smooth_layer1, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.base_model.smooth_layer2, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.base_model.smooth_layer3, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.rpn_net.rpn_conv_trans, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.rpn_net.rpn_conv_cls, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.rpn_net.rpn_conv_loc, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.top_model, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.fc_cls, 0, 0.01)
			fasterRCNNFPNBase.initWeights(self.fc_loc, 0, 0.001)
	'''random normal'''
	@staticmethod
	def initWeights(m, mean, stddev, bn_mean=1, bn_stddev=0.01):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			m.weight.data.normal_(mean, stddev)
			m.bias.data.fill_(0)
		elif classname.find('BatchNorm') != -1:
			m.weight.data.normal_(bn_mean, bn_stddev)
			m.bias.data.fill_(0)
	'''affine grid gen'''
	@staticmethod
	def affineGridGen(rois, input_size, grid_size, feature_stride=16):
		rois = rois.detach()
		x1 = rois[:, 1::4] / feature_stride
		y1 = rois[:, 2::4] / feature_stride
		x2 = rois[:, 3::4] / feature_stride
		y2 = rois[:, 4::4] / feature_stride
		height, width = input_size
		zero = rois.data.new(rois.size(0), 1).zero_()
		theta = torch.cat([(x2-x1)/(width-1), zero, (x1+x2-width+1)/(width-1), zero, (y2-y1)/(height-1), (y1+y2-height+1)/(height-1)], 1).view(-1, 2, 3)
		grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))
		return grid
	'''set bn fixed'''
	@staticmethod
	def setBnFixed(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			for p in m.parameters():
				p.requires_grad = False
	'''set bn eval'''
	@staticmethod
	def setBnEval(m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm') != -1:
			m.eval()


'''faster rcnn using resnet-FPN backbones'''
class FasterRCNNFPNResNets(fasterRCNNFPNBase):
	rpn_feature_strides = [4, 8, 16, 32, 64]
	rcnn_feature_strides = [4, 8, 16, 32]
	def __init__(self, mode, cfg, logger_handle, **kwargs):
		fasterRCNNFPNBase.__init__(self, cfg.NUM_CLASSES, cfg.IS_CLASS_AGNOSTIC, FasterRCNNFPNResNets.rpn_feature_strides, FasterRCNNFPNResNets.rcnn_feature_strides, mode, cfg)
		# base model
		self.base_model = FPNResNets(mode=mode, cfg=cfg, logger_handle=logger_handle)
		# RPN
		self.rpn_net = RegionProposalNet(in_channels=256, feature_strides=self.rpn_feature_strides, mode=mode, cfg=cfg)
		self.roi_crop = RoICrop()
		pooling_size = cfg.TRAIN_POOLING_SIZE if mode == 'TRAIN' else cfg.TEST_POOLING_SIZE
		self.roi_align = RoIAlignAvg(pooling_size, pooling_size)
		self.roi_pooling = RoIPooling(pooling_size, pooling_size)
		self.build_proposal_target_layer = buildProposalTargetLayer(mode, cfg)
		# define top model
		self.top_model = nn.Sequential(nn.Conv2d(256, 1024, kernel_size=pooling_size, stride=1, padding=0),
									   nn.ReLU(inplace=True),
									   nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
									   nn.ReLU(inplace=True))
		# final results
		self.fc_cls = nn.Linear(1024, self.num_classes)
		if self.is_class_agnostic:
			self.fc_loc = nn.Linear(1024, 4)
		else:
			self.fc_loc = nn.Linear(1024, 4*self.num_classes)
		if cfg.WEIGHTS_NEED_INITIALIZE and mode == 'TRAIN':
			self.initializeAddedModules()
		# fix some first layers following original implementation
		if cfg.FIXED_FRONT_BLOCKS:
			for p in self.base_model.base_layer0.parameters():
				p.requires_grad = False
			for p in self.base_model.base_layer1.parameters():
				p.requires_grad = False
		self.base_model.apply(fasterRCNNFPNBase.setBnFixed)
		self.top_model.apply(fasterRCNNFPNBase.setBnFixed)
	'''set train mode'''
	def setTrain(self):
		nn.Module.train(self, True)
		if self.cfg.FIXED_FRONT_BLOCKS:
			self.base_model.eval()
			self.base_model.base_layer2.train()
			self.base_model.base_layer3.train()
			self.base_model.base_layer4.train()
			self.base_model.lateral_layer0.train()
			self.base_model.lateral_layer1.train()
			self.base_model.lateral_layer2.train()
			self.base_model.lateral_layer3.train()
			self.base_model.smooth_layer1.train()
			self.base_model.smooth_layer2.train()
			self.base_model.smooth_layer3.train()
		self.base_model.apply(fasterRCNNFPNBase.setBnEval)
		self.top_model.apply(fasterRCNNFPNBase.setBnEval)