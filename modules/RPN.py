'''
Function:
	region proposal net
Author:
	Charles
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import *
from modules.losses import *
from libs.nms.nms_wrapper import nms


'''
Function:
	define the proposal layer for rpn
Init Input:
	--feature_strides: strides now.
	--mode: flag about TRAIN or TEST.
	--cfg: config file.
Forward Input:
	--probs_list: [(N, H*W, 2), ...]
	--x_reg_list: [(N, H*W, 4), ...]
	--rpn_features_shapes: a list for recording shapes of feature maps in each pyramid level
	--img_info: (height, width, scale_factor)
'''
class rpnProposalLayer(nn.Module):
	def __init__(self, feature_strides, mode, cfg, **kwargs):
		super(rpnProposalLayer, self).__init__()
		self.feature_strides = feature_strides
		self.anchor_generators = [AnchorGenerator(size_base=size_base, scales=cfg.ANCHOR_SCALES, ratios=cfg.ANCHOR_RATIOS) for size_base in cfg.ANCHOR_SIZE_BASES]
		if mode == 'TRAIN':
			self.pre_nms_topN = cfg.TRAIN_RPN_PRE_NMS_TOP_N
			self.post_nms_topN = cfg.TRAIN_RPN_POST_NMS_TOP_N
			self.nms_thresh = cfg.TRAIN_RPN_NMS_THRESH
		elif mode == 'TEST':
			self.pre_nms_topN = cfg.TEST_RPN_PRE_NMS_TOP_N
			self.post_nms_topN = cfg.TEST_RPN_POST_NMS_TOP_N
			self.nms_thresh = cfg.TEST_RPN_NMS_THRESH	
		else:
			raise ValueError('Unkown mode <%s> in rpnProposalLayer...' % mode)
	def forward(self, x):
		# parse x
		probs_list, x_reg_list, rpn_features_shapes, img_info = x
		# obtain proposals
		batch_size = probs_list[0].size(0)
		outputs = probs_list[0].new(batch_size, self.post_nms_topN, 5).zero_()
		for i in range(batch_size):
			output = []
			for probs, x_reg, rpn_features_shape, anchor_generator, feature_stride in zip(probs_list, x_reg_list, rpn_features_shapes, self.anchor_generators, self.feature_strides):
				# get bg and fg probs
				bg_probs = probs[i, :, 0]
				fg_probs = probs[i, :, 1]
				# get anchors
				anchors = anchor_generator.generate(feature_shape=rpn_features_shape, feature_stride=feature_stride, device=fg_probs.device).type_as(fg_probs)
				num_anchors = anchors.size(0)
				anchors = anchors.view(1, num_anchors, 4)
				# format x_reg
				bbox_deltas = x_reg[i:i+1, ...]
				# convert anchors to proposals
				proposals = BBoxFunctions.anchors2Proposals(anchors, bbox_deltas)
				# clip predicted boxes to image
				proposals = BBoxFunctions.clipBoxes(proposals, img_info[i:i+1, ...])
				# do nms
				proposals = proposals[0]
				scores = fg_probs
				_, order = torch.sort(scores, 0, True)
				if self.pre_nms_topN > 0 and self.pre_nms_topN < scores.numel():
					order = order[:self.pre_nms_topN]
				proposals = proposals[order]
				scores = scores[order].view(-1, 1)
				output.append(torch.cat((proposals, scores), 1))
			output = torch.cat(output)
			_, order = torch.sort(output[:, 4], 0, True)
			output = output[order]
			_, keep_idxs = nms(output, self.nms_thresh)
			keep_idxs = keep_idxs.long().view(-1)
			if self.post_nms_topN > 0:
				keep_idxs = keep_idxs[:self.post_nms_topN]
			proposals = output[keep_idxs, :4]
			num_proposals = proposals.size(0)
			outputs[i, :, 0] = i
			outputs[i, :num_proposals, 1:] = proposals
		return outputs
	def backward(self, *args):
		pass


'''build target layer for rpn'''
class rpnBuildTargetLayer(nn.Module):
	def __init__(self, feature_strides, mode, cfg, **kwargs):
		super(rpnBuildTargetLayer, self).__init__()
		self.feature_strides = feature_strides
		self.anchor_generators = [AnchorGenerator(size_base=size_base, scales=cfg.ANCHOR_SCALES, ratios=cfg.ANCHOR_RATIOS) for size_base in cfg.ANCHOR_SIZE_BASES]
		if mode == 'TRAIN':
			self.rpn_negative_overlap = cfg.TRAIN_RPN_NEGATIVE_OVERLAP
			self.rpn_positive_overlap = cfg.TRAIN_RPN_POSITIVE_OVERLAP
			self.rpn_fg_fraction = cfg.TRAIN_RPN_FG_FRACTION
			self.rpn_batch_size = cfg.TRAIN_RPN_BATCHSIZE
		elif mode == 'TEST':
			self.rpn_negative_overlap = cfg.TEST_RPN_NEGATIVE_OVERLAP
			self.rpn_positive_overlap = cfg.TEST_RPN_POSITIVE_OVERLAP
			self.rpn_fg_fraction = cfg.TEST_RPN_FG_FRACTION
			self.rpn_batch_size = cfg.TEST_RPN_BATCHSIZE
		else:
			raise ValueError('Unkown mode <%s> in rpnBuildTargetLayer...' % mode)
		self.allowed_border = 0
	def forward(self, x):
		# parse x
		gt_boxes, rpn_features_shapes, img_info, num_gt_boxes = x
		batch_size = gt_boxes.size(0)
		# get anchors
		anchors = []
		for rpn_features_shape, anchor_generator, feature_stride in zip(rpn_features_shapes, self.anchor_generators, self.feature_strides):
			anchors.append(anchor_generator.generate(feature_shape=rpn_features_shape, feature_stride=feature_stride, device=gt_boxes.device))
		anchors = torch.cat(anchors, 0).type_as(gt_boxes)
		total_anchors_ori = anchors.size(0)
		# make sure anchors are in the image
		keep_idxs = ((anchors[:, 0] >= -self.allowed_border) &
					 (anchors[:, 1] >= -self.allowed_border) &
					 (anchors[:, 2] < int(img_info[0][1])+self.allowed_border) &
					 (anchors[:, 3] < int(img_info[0][0])+self.allowed_border))
		keep_idxs = torch.nonzero(keep_idxs).view(-1)
		anchors = anchors[keep_idxs, :]
		# prepare labels: 1 is positive, 0 is negative, -1 means ignore
		labels = gt_boxes.new(batch_size, keep_idxs.size(0)).fill_(-1)
		# calc ious
		overlaps = BBoxFunctions.calcIoUs(anchors, gt_boxes)
		max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
		gt_max_overlaps, _ = torch.max(overlaps, 1)
		# assign labels
		labels[max_overlaps < self.rpn_negative_overlap] = 0
		gt_max_overlaps[gt_max_overlaps==0] = 1e-5
		keep_idxs_label = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)
		if torch.sum(keep_idxs_label) > 0:
			labels[keep_idxs_label > 0] = 1
		labels[max_overlaps >= self.rpn_positive_overlap] = 1
		max_num_fg = int(self.rpn_fg_fraction * self.rpn_batch_size)
		num_fg = torch.sum((labels == 1).int(), 1)
		num_bg = torch.sum((labels == 0).int(), 1)
		for i in range(batch_size):
			if num_fg[i] > max_num_fg:
				fg_idxs = torch.nonzero(labels[i] == 1).view(-1)
				rand_num = torch.from_numpy(np.random.permutation(fg_idxs.size(0))).type_as(gt_boxes).long()
				disable_idxs = fg_idxs[rand_num[:fg_idxs.size(0)-max_num_fg]]
				labels[i][disable_idxs] = -1
			max_num_bg = self.rpn_batch_size - torch.sum((labels == 1).int(), 1)[i]
			if num_bg[i] > max_num_bg:
				bg_idxs = torch.nonzero(labels[i] == 0).view(-1)
				rand_num = torch.from_numpy(np.random.permutation(bg_idxs.size(0))).type_as(gt_boxes).long()
				disable_idxs = bg_idxs[rand_num[:bg_idxs.size(0)-max_num_bg]]
				labels[i][disable_idxs] = -1
		offsets = torch.arange(0, batch_size) * gt_boxes.size(1)
		argmax_overlaps = argmax_overlaps + offsets.view(batch_size, 1).type_as(argmax_overlaps)
		gt_rois = gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5)
		bbox_targets = BBoxFunctions.encodeBboxes(anchors, gt_rois[..., :4])
		# unmap
		labels = rpnBuildTargetLayer.unmap(labels, total_anchors_ori, keep_idxs, batch_size, fill=-1)
		bbox_targets = rpnBuildTargetLayer.unmap(bbox_targets, total_anchors_ori, keep_idxs, batch_size, fill=0)
		# pack return values into outputs
		outputs = [labels, bbox_targets]
		return outputs
	@staticmethod
	def unmap(data, count, inds, batch_size, fill=0):
		if data.dim() == 2:
			ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
			ret[:, inds] = data
		else:
			ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
			ret[:, inds, :] = data
		return ret
	def backward(self, *args):
		pass


'''region proposal net'''
class RegionProposalNet(nn.Module):
	def __init__(self, in_channels, feature_strides, mode, cfg, **kwargs):
		super(RegionProposalNet, self).__init__()
		# prepare
		self.mode = mode
		self.cfg = cfg
		# define rpn conv
		self.rpn_conv_trans = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
											nn.ReLU(inplace=True))
		self.out_channels_cls = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS) * 2
		self.out_channels_reg = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS) * 4
		self.rpn_conv_cls = nn.Conv2d(in_channels=512, out_channels=self.out_channels_cls, kernel_size=1, stride=1, padding=0)
		self.rpn_conv_reg = nn.Conv2d(in_channels=512, out_channels=self.out_channels_reg, kernel_size=1, stride=1, padding=0)
		# proposal layer
		self.rpn_proposal_layer = rpnProposalLayer(feature_strides=feature_strides, mode=mode, cfg=cfg)
		# build target layer
		self.rpn_build_target_layer = rpnBuildTargetLayer(feature_strides=feature_strides, mode=mode, cfg=cfg)
	'''forward'''
	def forward(self, rpn_features, gt_boxes, img_info, num_gt_boxes):
		batch_size = rpn_features[0].size(0)
		# get all predict results
		rpn_features_shapes = []
		x_cls_list = []
		x_reg_list = []
		probs_list = []
		for i in range(len(rpn_features)):
			x = rpn_features[i]
			rpn_features_shapes.append([x.size(2), x.size(3)])
			# --do base classifiction and regression
			x = self.rpn_conv_trans(x)
			x_cls = self.rpn_conv_cls(x)
			x_reg = self.rpn_conv_reg(x)
			# --do softmax to get probs
			x_cls_reshape = x_cls.view(x_cls.size(0), 2, -1, x_cls.size(3))
			probs = F.softmax(x_cls_reshape, 1)
			probs = probs.view(x_cls.size())
			# --format results
			x_cls_list.append(x_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
			probs_list.append(probs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
			x_reg_list.append(x_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))
		x_cls_concat = torch.cat(x_cls_list, axis=1)
		x_reg_concat = torch.cat(x_reg_list, axis=1)
		# get RoIs
		rois = self.rpn_proposal_layer(([p.data for p in probs_list], [x_reg.data for x_reg in x_reg_list], rpn_features_shapes, img_info))
		# define loss
		rpn_cls_loss = torch.Tensor([0]).type_as(x)
		rpn_reg_loss = torch.Tensor([0]).type_as(x)
		# while training, calculate loss
		if self.mode == 'TRAIN' and gt_boxes is not None:
			targets = self.rpn_build_target_layer((gt_boxes, rpn_features_shapes, img_info, num_gt_boxes))
			# --classification loss
			labels = targets[0].view(batch_size, -1)
			keep_idxs = labels.view(-1).ne(-1).nonzero().view(-1)
			x_cls_concat_keep = torch.index_select(x_cls_concat.view(-1, 2), 0, keep_idxs.data)
			labels_keep = torch.index_select(labels.view(-1), 0, keep_idxs.data)
			labels_keep = labels_keep.long()
			if self.cfg.RPN_CLS_LOSS_SET['type'] == 'cross_entropy':
				rpn_cls_loss = F.cross_entropy(x_cls_concat_keep, labels_keep, size_average=self.cfg.RPN_CLS_LOSS_SET['cross_entropy']['size_average'])
				rpn_cls_loss = rpn_cls_loss * self.cfg.RPN_CLS_LOSS_SET['cross_entropy']['weight']
			else:
				raise ValueError('Unkown classification loss type <%s>...' % self.cfg.RPN_CLS_LOSS_SET['type'])
			# --regression loss
			bbox_targets = targets[1]
			if self.cfg.RPN_REG_LOSS_SET['type'] == 'betaSmoothL1Loss':
				mask = targets[0].unsqueeze(2).expand(batch_size, targets[0].size(1), 4)
				rpn_reg_loss = betaSmoothL1Loss(x_reg_concat[mask>0].view(-1, 4), 
												bbox_targets[mask>0].view(-1, 4), 
												beta=self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['beta'], 
												size_average=self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['size_average'],
												loss_weight=self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['weight'],
												avg_factor=x_reg_concat[mask>0].view(-1, 4).size(0))
			else:
				raise ValueError('Unkown regression loss type <%s>...' % self.cfg.RPN_REG_LOSS_SET['type'])
		return rois, rpn_cls_loss, rpn_reg_loss
	'''initialize weights'''
	def initWeights(self, init_method):
		# normal init
		if init_method == 'normal':
			for layer in [self.rpn_conv_trans[0], self.rpn_conv_cls, self.rpn_conv_reg]:
				normalInit(layer, std=0.01)
		# kaiming init
		elif init_method == 'kaiming':
			for layer in [self.rpn_conv_trans[0], self.rpn_conv_cls, self.rpn_conv_reg]:
				kaimingInit(layer, nonlinearity='relu')
		# xavier
		elif init_method == 'xavier':
			for layer in [self.rpn_conv_trans[0], self.rpn_conv_cls, self.rpn_conv_reg]:
				xavierInit(layer, distribution='uniform')
		# unsupport
		else:
			raise RuntimeError('Unsupport initializeAddedModules.init_method <%s>...' % init_method)