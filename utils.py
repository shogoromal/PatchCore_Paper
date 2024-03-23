import os
import numpy as np
import cv2
import torch
from torch.nn import functional as F
import pytorch_lightning as pl

def embedding_concat(x, y):

  B, C1, H1, W1 = x.size()
  _, C2, H2, W2 = y.size()

  s = int(H1/H2)
  x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
  x = x.view(B, C1, -1, H2, W2)
  z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)

  for i in range(x.size(2)):
    z[:,:,i,:,:] = torch.cat((x[:,:,i, :,:], y), 1)
  z = z.view(B, -1, H2 * W2)
  z = F.fold(z, kernel_size=s, output_size=(H1,W1), stride=s)

  return z

def reshape_embedding(embedding):
  embedding_list = []
  for k in range(embedding.shape[0]):
    for i in range(embedding.shape[2]):
      for j in range(embedding.shape[3]):
        embedding_list.append(embedding[k, :, i, j])

  return embedding_list

def prep_dirs(root):
    # make embeddings dir
    embeddings_path = os.path.join(root, 'tmp', 'temp_embedding')
    os.makedirs(embeddings_path, exist_ok=True)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    heatmap = heatmap[:,:,[2,1,0]]#表示の関係でRGBを入れ替える
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)