from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from os.path import join
import torch
import numpy as np
from sklearn import preprocessing

Scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

dataindex = 151

class MMIDataset(Dataset):

    def __init__(self, img_size, z_dim, points_path, img_folder,points_original, dataindex):
        self.data = pd.read_csv(points_path, header=0, index_col=None).to_numpy()
        # self.data = pd.read_csv(points_path, header=0).to_numpy()
        self.img_folder = img_folder
        self.img_size = img_size
        self.z_dim = z_dim
        self.points_original = pd.read_csv(points_original, header=None, index_col=None).to_numpy()
        self.dataindex = dataindex

    def __getitem__(self, index):
        item = self.data[index]
        img = cv2.imread(self.img_folder + '\\{}.png'.format(item[0]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))[:, :, np.newaxis]
        img = img / 255.0 * 2 - 1
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        
        points21 = item[1:self.dataindex].astype(np.float64).reshape(-1, 1)
        points21 = Scaler.fit_transform(points21)
        points21 = torch.from_numpy(points21).flatten(0)
       
        points = item[1:self.dataindex].astype(np.float64).reshape(-1,1)
        points = Scaler.fit_transform(points)
        points = torch.from_numpy(points).flatten(0)
        points = points.reshape([len(points), 1, 1])
        # the shape of points should be [Z_DIM, CHANNELS_IMG, FEATURES_GEN]

        points_original_1 = self.points_original[index]
        points_original_2 = points_original_1[1:22].astype(np.float64).reshape(-1,1)
        points_original = torch.from_numpy(points_original_2).flatten(0)

        return points, img, points21, points_original

    def __len__(self):
        return len(self.data)


def get_loader(
        img_size,
        batch_size,
        z_dim,
        points_path='C:/Users/Administrator',
        img_folder='C:/Users/Administrator/',
        points_original = 'C:/Users/Administrator/',
        shuffle=True,
        dataindex = 151
):
    return DataLoader(MMIDataset(img_size, z_dim, points_path, img_folder, points_original,dataindex),
                      batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    pass
