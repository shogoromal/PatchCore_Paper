from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
  def __init__(self, img_set, transform):
      self.img_set = img_set
      self.transform = transform

  def __len__(self):
    return len(self.img_set)

  def __getitem__(self, idx):
    img = self.img_set[idx]
    img = Image.fromarray(img)
    img = self.transform(img)
    return img