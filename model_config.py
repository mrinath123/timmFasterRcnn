import torch
from torch.utils.data import Dataset, DataLoader

effnet_b3 = {
"feat_size":[32, 48, 136, 384],
"return_layers":{'1':'0', '2': '1', '4': '2', '6': '3'}
}

class BackBoneFeats(Dataset):
    def __init__(self,name):
        super().__init__()
        self.name = name
        if (self.name == "effnet_b3"):
                self.feat_size = effnet_b3["feat_size"]
                self.return_layers = effnet_b3["return_layers"]
        else:
            raise Exception('no backbone of this name is present')



if __name__ == "__main__":
    bf = BackBoneFeats("effnet_3")
    print(bf.return_layers)
    print(bf.feat_size)


