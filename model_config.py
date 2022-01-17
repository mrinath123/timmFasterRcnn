import timm
from torch.utils.data import Dataset

effnet_b3 = {
"feat_size":[32, 48, 136, 384],
"return_layers":{'1':'0', '2': '1', '4': '2', '6': '3'}
}

effnet_b4 = {
"feat_size":[32, 56, 160, 448],
"return_layers":{'1':'0', '2': '1', '4': '2', '6': '3'}
}

effnet_b5 = {
"feat_size":[40, 64, 176, 512],
"return_layers":{'1':'0', '2': '1', '4': '2', '6': '3'}
}

effnet_b6 = {
"feat_size":[40, 72, 200, 576],
"return_layers":{'1':'0', '2': '1', '4': '2', '6': '3'}
}

effnet_b7 = {
"feat_size":[48, 80, 224, 640],
"return_layers":{'1':'0', '2': '1', '4': '2', '6': '3'}
}

densenet_121 = {
"feat_size":[256, 512, 1024, 1024],
"return_layers":{'denseblock1':'0', 'denseblock2': '1', 'denseblock3': '2', 'denseblock4': '3'}
}


class BackBoneFeats(Dataset):
    def __init__(self,name,pretrained = False):
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        if (self.name == "efficientnet_b3"):
            self.backbone = timm.create_model(self.name, pretrained=self.pretrained,num_classes=0, global_pool='')
            self.feat_size = effnet_b3["feat_size"]
            self.return_layers = effnet_b3["return_layers"]
            self.out_channels = self.feat_size[len(self.feat_size) -1]

        elif (self.name == "efficientnet_b4"):
            self.backbone = timm.create_model(self.name, pretrained=self.pretrained,num_classes=0, global_pool='')
            self.feat_size = effnet_b4["feat_size"]
            self.return_layers = effnet_b4["return_layers"]
            self.out_channels = self.feat_size[len(self.feat_size) -1]

        elif (self.name == "efficientnet_b5"):
            self.backbone = timm.create_model(self.name, pretrained=self.pretrained,num_classes=0, global_pool='')
            self.feat_size = effnet_b5["feat_size"]
            self.return_layers = effnet_b5["return_layers"]
            self.out_channels = self.feat_size[len(self.feat_size) -1]

        elif (self.name == "efficientnet_b6"):
            self.backbone = timm.create_model(self.name, pretrained=self.pretrained,num_classes=0, global_pool='')
            self.feat_size = effnet_b6["feat_size"]
            self.return_layers = effnet_b6["return_layers"]
            self.out_channels = self.feat_size[len(self.feat_size) -1]

        elif (self.name == "efficientnet_b7"):
            self.backbone = timm.create_model(self.name, pretrained=self.pretrained,num_classes=0, global_pool='')
            self.feat_size = effnet_b7["feat_size"]
            self.return_layers = effnet_b7["return_layers"]
            self.out_channels = self.feat_size[len(self.feat_size) -1]

        elif(self.name == "densenet121"):
            self.backbone = timm.create_model(self.name, pretrained=self.pretrained,num_classes=0, global_pool='')
            self.feat_size = densenet_121["feat_size"]
            self.return_layers = densenet_121["return_layers"]
            self.out_channels = self.feat_size[len(self.feat_size) -1]

        else:
            raise Exception('no backbone of this name is present')



if __name__ == "__main__":
    bf = BackBoneFeats("efficientnet_b3")
    print(bf.return_layers)
    print(bf.feat_size)
    print(bf.out_channels)


