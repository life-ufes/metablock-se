from models.effnet import MyEffnet
from models.mobilenet import MyMobilenet
from models.resnet import MyResnet
from models.timmmodel import GenericTimmModel
from torchvision import models
from efficientnet_pytorch import EfficientNet

CONFIG_METABLOCK_BY_MODEL = {
    'caformer_s18': 16, 
    'resnet-50': 64,
    'mobilenet': 40,
    'efficientnet-b4': 56,
}

_DEFAULT_WEIGHTS = {
    'resnet-50': 'ResNet50_Weights.DEFAULT',
    'mobilenet': 'MobileNet_V2_Weights.DEFAULT',    
    'efficientnet-b4': True,
    'caformer_s18': {
        'weights': 'caformer_s18.sail_in1k',
        'n_feat_conv': 512,
    },
}

_MODELS = ['resnet-50', 'mobilenet', 'efficientnet-b4', 'caformer_s18']

def set_class_model (model_name, num_class, neurons_reducer_block=0, comb_method=None, 
                     comb_config=None, pretrained=True, freeze_conv=False, initial_weights_path = None):

    if pretrained:
        pre_torch = _DEFAULT_WEIGHTS[model_name]
    else:
        pre_torch = None

    if model_name not in _MODELS:
        raise Exception(f"The model {model_name} is not available!")

    model = None
    if model_name == 'resnet-50':
        model = MyResnet(models.resnet50(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'mobilenet':
        model = MyMobilenet(models.mobilenet_v2(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)
    elif model_name  == 'efficientnet-b4':
        if pretrained:
            model = MyEffnet(EfficientNet.from_pretrained(model_name), num_class, neurons_reducer_block, freeze_conv,
                             comb_method=comb_method, comb_config=comb_config)
        else:
            model = MyEffnet(EfficientNet.from_name(model_name), num_class, neurons_reducer_block, freeze_conv,
                             comb_method=comb_method, comb_config=comb_config)
    elif _DEFAULT_WEIGHTS[model_name]:
        model = GenericTimmModel(_DEFAULT_WEIGHTS[model_name]['weights'], num_class, neurons_reducer_block, freeze_conv, comb_method=comb_method, comb_config=comb_config,
                                 n_feat_conv = _DEFAULT_WEIGHTS[model_name]['n_feat_conv'], initial_weights_path=initial_weights_path)
                             
    return model

