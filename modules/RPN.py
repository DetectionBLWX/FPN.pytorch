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
from libs.nms.nms_wrapper import nms
from modules.utils.utils import BBoxFunctions
from modules.losses.smoothL1 import betaSmoothL1Loss


'''
Function:
	define the proposal layer for rpn
Init Input:
	--feature_strides: strides now.
	--anchor_scales: scales for anchor boxes
	--anchor_ratios: ratios for anchor boxes
	--mode: flag about TRAIN or TEST.
	--cfg: config file.
Forward Input:
	--x_cls_pred/probs: (N, H*W, 2)
	--x_loc_pred: (N, H*W, 4)
	--rpn_features_shapes: a list for recording shapes of feature maps in each pyramid level
	--img_info: (height, width, scale_factor)
'''
class rpnProposalLayer(nn.Module):
	def __init__(self, feature_strides, anchor_scales, anchor_ratios, mode, cfg, **kwargs):
		super(rpnProposalLayer, self).__init__()
		self.feature_strides = feature_strides
		self.anchor_scales = anchor_scales
		self.anchor_ratios = anchor_ratios
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
		# prepare
		probs, x_loc_pred, rpn_features_shapes, img_info = x
		batch_size = probs.size(0)
		# get bg and fg probs
		bg_probs = probs[..., 0]
		fg_probs = probs[..., 1]
		# get anchors
		anchors = RegionProposalNet.generateAnchors(scales=self.anchor_scales, ratios=self.anchor_ratios, feature_shapes=rpn_features_shapes, feature_strides=self.feature_strides).type_as(fg_probs)
		num_anchors = anchors.size(0)
		anchors = anchors.view(1, num_anchors, 4).expand(batch_size, num_anchors, 4)
		# format x_loc_pred
		bbox_deltas = x_loc_pred
		# convert anchors to proposals
		proposals = BBoxFunctions.anchors2Proposals(anchors, bbox_deltas)
		# clip predicted boxes to image
		proposals = BBoxFunctions.clipBoxes(proposals, img_info)
		# do nms
		scores = fg_probs
		_, order = torch.sort(scores, 1, True)
		output = scores.new(batch_size, self.post_nms_topN, 5).zero_()
		for i in range(batch_size):
			proposals_single = proposals[i]
			scores_single = scores[i]
			order_single = order[i]
			if self.pre_nms_topN > 0 and self.pre_nms_topN < scores.numel():
				order_single = order_single[:self.pre_nms_topN]
			proposals_single = proposals_single[order_single, :]
			scores_single = scores_single[order_single].view(-1, 1)
			_, keep_idxs = nms(torch.cat((proposals_single, scores_single), 1), self.nms_thresh)
			keep_idxs = keep_idxs.long().view(-1)
			if self.post_nms_topN > 0:
				keep_idxs = keep_idxs[:self.post_nms_topN]
			proposals_single = proposals_single[keep_idxs, :]
			scores_single = scores_single[keep_idxs, :]
			num_proposals = proposals_single.size(0)
			output[i, :, 0] = i
			output[i, :num_proposals, 1:] = proposals_single
		return output
	def backward(self, *args):
		pass


'''build target layer for rpn'''
class rpnBuildTargetLayer(nn.Module):
	def __init__(self, feature_strides, anchor_scales, anchor_ratios, mode, cfg, **kwargs):
		super(rpnBuildTargetLayer, self).__init__()
		self.feature_strides = feature_strides
		self.anchor_scales = anchor_scales
		self.anchor_ratios = anchor_ratios
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
		# prepare
		x_cls_pred, gt_boxes, rpn_features_shapes, img_info, num_gt_boxes = x
		batch_size = gt_boxes.size(0)
		# get anchors
		anchors = RegionProposalNet.generateAnchors(scales=self.anchor_scales, ratios=self.anchor_ratios, feature_shapes=rpn_features_shapes, feature_strides=self.feature_strides).type_as(x_cls_pred)
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
		self.anchor_scales = cfg.ANCHOR_SCALES
		self.anchor_ratios = cfg.ANCHOR_RATIOS
		self.feature_strides = feature_strides
		self.in_channels = in_channels
		self.mode = mode
		self.cfg = cfg
		# define rpn conv
		self.rpn_conv_trans = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
											nn.ReLU(inplace=True))
		self.out_channels_cls = 1 * len(self.anchor_ratios) * 2
		self.out_channels_loc = 1 * len(self.anchor_ratios) * 4
		self.rpn_conv_cls = nn.Conv2d(in_channels=512, out_channels=self.out_channels_cls, kernel_size=1, stride=1, padding=0)
		self.rpn_conv_loc = nn.Conv2d(in_channels=512, out_channels=self.out_channels_loc, kernel_size=1, stride=1, padding=0)
		# proposal layer
		self.rpn_proposal_layer = rpnProposalLayer(feature_strides=self.feature_strides, anchor_scales=self.anchor_scales, anchor_ratios=self.anchor_ratios, mode=self.mode, cfg=self.cfg)
		# build target layer
		self.rpn_build_target_layer = rpnBuildTargetLayer(feature_strides=self.feature_strides, anchor_scales=self.anchor_scales, anchor_ratios=self.anchor_ratios, mode=self.mode, cfg=self.cfg)
	def forward(self, rpn_features, gt_boxes, img_info, num_gt_boxes):
		batch_size = rpn_features[0].size(0)
		# get all predict results
		rpn_features_shapes = []
		x_cls_list = []
		x_loc_list = []
		probs_list = []
		for i in range(len(rpn_features)):
			x = rpn_features[i]
			rpn_features_shapes.append([x.size(2), x.size(3)])
			# --do base classifiction and location
			x = self.rpn_conv_trans(x)
			x_cls = self.rpn_conv_cls(x)
			x_loc = self.rpn_conv_loc(x)
			# --do softmax to get probs
			x_cls_reshape = x_cls.view(x_cls.size(0), 2, -1, x_cls.size(3))
			probs = F.softmax(x_cls_reshape, 1)
			probs = probs.view(x_cls.size())
			# --format results
			x_cls_list.append(x_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
			probs_list.append(probs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
			x_loc_list.append(x_loc.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))
		x_cls_all = torch.cat(x_cls_list, 1)
		probs_all = torch.cat(probs_list, 1)
		x_loc_all = torch.cat(x_loc_list, 1)
		# get RoIs
		rois = self.rpn_proposal_layer((probs_all.data, x_loc_all.data, rpn_features_shapes, img_info))
		# define loss
		rpn_cls_loss = torch.Tensor([0]).type_as(x)
		rpn_loc_loss = torch.Tensor([0]).type_as(x)
		# while training, calculate loss
		if self.mode == 'TRAIN' and gt_boxes is not None:
			targets = self.rpn_build_target_layer((x_cls_all.data, gt_boxes, rpn_features_shapes, img_info, num_gt_boxes))
			# --classification loss
			labels = targets[0].view(batch_size, -1)
			keep_idxs = labels.view(-1).ne(-1).nonzero().view(-1)
			cls_scores_pred_keep = torch.index_select(x_cls_all.view(-1, 2), 0, keep_idxs.data)
			labels_keep = torch.index_select(labels.view(-1), 0, keep_idxs.data)
			labels_keep = labels_keep.long()
			if self.cfg.RPN_CLS_LOSS_SET['type'] == 'cross_entropy':
				rpn_cls_loss = F.cross_entropy(cls_scores_pred_keep, labels_keep, size_average=self.cfg.RPN_CLS_LOSS_SET['cross_entropy']['size_average'])
				rpn_cls_loss = rpn_cls_loss * self.cfg.RPN_CLS_LOSS_SET['cross_entropy']['weight']
			else:
				raise ValueError('Unkown classification loss type <%s>...' % self.cfg.RPN_CLS_LOSS_SET['type'])
			# --regression loss
			bbox_targets = targets[1]
			if self.cfg.RPN_REG_LOSS_SET['type'] == 'betaSmoothL1Loss':
				mask = targets[0].unsqueeze(2).expand(batch_size, targets[0].size(1), 4)
				rpn_loc_loss = betaSmoothL1Loss(x_loc_all[mask>0].view(-1, 4), 
												bbox_targets[mask>0].view(-1, 4), 
												beta=self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['beta'], 
												size_average=self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['size_average'],
												loss_weight=self.cfg.RPN_REG_LOSS_SET['betaSmoothL1Loss']['weight'])
			else:
				raise ValueError('Unkown regression loss type <%s>...' % self.cfg.RPN_REG_LOSS_SET['type'])
		return rois, rpn_cls_loss, rpn_loc_loss
	'''
	Function:
		generate anchors.
	Input:
		--base_size(int): the base anchor size (8 in FPN).
		--scales(list): scales for each pyramid level.
		--ratios(list): ratios for anchor boxes.
		--feature_shapes(list): the size of feature maps in each pyramid level.
		--feature_strides(list): the strides in each pyramid level.
	Return:
		--anchors(np.array): [nA, 4], the format is (x1, y1, x2, y2).
	'''
	@staticmethod
	def generateAnchors(size_base=8, scales=2**np.arange(2, 7), ratios=[0.5, 1, 2], feature_shapes=list(), feature_strides=list()):
		assert (len(scales) == len(feature_shapes)) and (len(feature_shapes) == len(feature_strides)), 'for <scales> <feature_shapes> and <feature_strides>, expect same length.'
		anchors = []
		for i in range(len(scales)):
			scales_pyramid, ratios_pyramid = np.meshgrid(np.array(scales[i]*size_base), np.array(ratios))
			scales_pyramid, ratios_pyramid = scales_pyramid.flatten(), ratios_pyramid.flatten()
			heights = scales_pyramid / np.sqrt(ratios_pyramid)
			widths = scales_pyramid * np.sqrt(ratios_pyramid)
			shifts_x = np.arange(0, feature_shapes[i][1], 1) * feature_strides[i] + 0.5 * feature_strides[i]
			shifts_y = np.arange(0, feature_shapes[i][0], 1) * feature_strides[i] + 0.5 * feature_strides[i]
			shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
			widths, cxs = np.meshgrid(widths, shifts_x)
			heights, cys = np.meshgrid(heights, shifts_y)
			boxes_cxcy = np.stack([cxs, cys], axis=2).reshape([-1, 2])
			boxes_whs = np.stack([widths, heights], axis=2).reshape([-1, 2])
			anchors_pyramid = np.concatenate([boxes_cxcy-0.5*boxes_whs, boxes_cxcy+0.5*boxes_whs], axis=1)
			anchors.append(anchors_pyramid)
		anchors = np.concatenate(anchors, axis=0)
		return torch.from_numpy(anchors).float()