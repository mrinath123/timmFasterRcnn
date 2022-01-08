import timm
from torch.utils.data import Dataset

effnet_b3 = {
"feat_size":[32, 48, 136, 384],
"return_layers":{'1':'0', '2': '1', '4': '2', '6': '3'}
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
        else:
            raise Exception('no backbone of this name is present')



if __name__ == "__main__":
    bf = BackBoneFeats("efficientnet_b3")
    print(bf.return_layers)
    print(bf.feat_size)
    print(bf.out_channels)


