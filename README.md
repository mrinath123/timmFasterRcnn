# timmFasterRcnn
Use timm pretrained backbones for you FasterRCNN!

1. model_config.py -> it returns the model,feat_sizes,output channel and the feat layer names, which is reqd by the Add_FPN.py file

2. Add_FPN.py -> Edited the BackboneWithFPN function from pytorch, which is now used to add FPN to any timm model, till now it can be only used for Efficentnet family

3. test.ipynb -> an example explaining how to use it any backbone and add FPN to it


