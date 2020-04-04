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
Input for __init__:
	--size_base(int): the base anchor size.
	--scales(list): scales for anchor boxes.
	--ratios(list): ratios for anchor boxes.
Input for generate:
	--feature_shape(tuple): the size of feature maps in corresponding pyramid level.
	--feature_stride(int): the feature stride in corresponding pyramid level.
	--device: specify cpu or cuda.
Return:
	--anchors(torch.FloatTensor): [nA, 4], the format is (x1, y1, x2, y2).
'''
class AnchorGenerator(object):
	def __init__(self, size_base, scales=[8], ratios=[0.5, 1, 2], **kwargs):
		self.size_base = size_base
		self.scales = torch.Tensor(scales)
		self.ratios = torch.Tensor(ratios)
		self.base_anchors = self.__generateBaseAnchors()
	'''generate anchors'''
	def generate(self, feature_shape=None, feature_stride=None, device='cuda'):
		base_anchors = self.base_anchors.to(device)
		feat_h, feat_w = feature_shape
		shift_x = torch.arange(0, feat_w, device=device) * feature_stride
		shift_y = torch.arange(0, feat_h, device=device) * feature_stride
		shift_xx, shift_yy = self.__meshgrid(shift_x, shift_y)
		shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
		shifts = shifts.type_as(base_anchors)
		all_anchors = base_anchors[None, :, :] + shifts[:, None, :].float()
		all_anchors = all_anchors.view(-1, 4)
		return all_anchors
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