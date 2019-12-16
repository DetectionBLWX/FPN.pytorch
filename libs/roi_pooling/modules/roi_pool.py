from torch.nn.modules.module import Module
from ..functions.roi_pool import RoIPoolFunction


class RoIPooling(Module):
    def __init__(self, pooled_height, pooled_width):
        super(RoIPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)

    def forward(self, features, rois, spatial_scale):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, float(spatial_scale))(features, rois)
