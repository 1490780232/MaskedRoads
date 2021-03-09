import models.Deeplab_v3_plus as network
from .utils import set_bn_momentum
def get_deeplabv3plus_resnet101(num_classes, output_stride=16, separable_conv=True, modelname = "deeplabv3plus_resnet101"):
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    model = model_map[modelname](num_classes=num_classes, output_stride=output_stride)
    if separable_conv and 'plus' in modelname:
        network.convert_to_separable_conv(model.classifier)
    set_bn_momentum(model.backbone, momentum=0.01)
    return model