'''
Function:
	define the utils to generate anchors
Author:
	Charles
'''
import torch


'''
Function:
	anchor generator
Input:
	--size_base(int): the base anchor size.
	--scales(list): scales for anchor boxes.
	--ratios(list): ratios for anchor boxes.
	--feature_shape(tuple): the size of feature maps in corresponding pyramid level.
	--feature_stride(int): the feature stride in corresponding pyramid level.
Return:
	--anchors(torch.FloatTensor): [nA, 4], the format is (x1, y1, x2, y2).
'''
class AnchorGenerator(object):
	def __init__(self, size_base, scales=[8], ratios=[0.5, 1, 2], feature_shape=None, feature_stride=None, **kwargs):
		self.size_base = size_base
		self.scales = scales
		self.ratios = ratios
		self.feature_shape = feature_shape
		self.feature_stride = feature_stride
	'''generate anchors'''
	def generate(self):
		base_anchors = self.__generateBaseAnchors()
		feat_h, feat_w = self.feature_shape
		shift_x = torch.arange(0, feat_w) * self.feature_stride
		shift_y = torch.arange(0, feat_h) * self.feature_stride
		shift_xx, shift_yy = self.__meshgrid(shift_x, shift_y)
		shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
		all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
		all_anchors = all_anchors.view(-1, 4)
		return all_anchors.float()
	'''meshgrid'''
	def __meshgrid(self, x, y):
		xx = x.repeat(len(y))
		yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
		return xx, yy
	'''generate base anchors'''
	def __generateBaseAnchors(self):
		w = self.size_base
		h = self.size_base
		x_ctr = 0.5 * (w - 1)
		y_ctr = 0.5 * (h - 1)
		h_ratios = torch.sqrt(self.ratios)
		w_ratios = 1 / h_ratios
		ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
		hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
		base_anchors = torch.stack([x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)], dim=-1).round()
		return base_anchors