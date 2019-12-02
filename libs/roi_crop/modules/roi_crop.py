from torch.nn.modules.module import Module
from ..functions.roi_crop import RoICropFunction

class RoICrop(Module):
    def __init__(self, layout = 'BHWD'):
        super(RoICrop, self).__init__()
    def forward(self, input1, input2):
        return RoICropFunction()(input1, input2)
