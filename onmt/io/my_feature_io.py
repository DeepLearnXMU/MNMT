import  torch 
from torch.utils.data import Dataset,DataLoader
import os
import tables



class custom_dset(Dataset):
    def __init__(self,
                features,idxs):
        self.features=features
        self.idxs=idxs
    
    
    def __getitem__(self, index):
        return self.features[self.idxs[index]]
    

    def __len__(self):
        return len(self.idxs)





    
        

