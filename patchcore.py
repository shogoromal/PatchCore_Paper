
import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import faiss
from utils import embedding_concat, reshape_embedding, prep_dirs
from dataset import MyDataset

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class PatchCore(pl.LightningModule):

  def __init__(self, hparams):

    super(PatchCore, self).__init__()
    self.save_hyperparameters(hparams)

    # PatchCore の別のメソッドで定義する
    self.init_features()
    self.max_Nb = 0

    def hook_t(module, input, output):
      self.features.append(output)

    self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

    for params in self.model.parameters():
      params.requires_grad = False

    self.model.layer2[-1].register_forward_hook(hook_t)
    self.model.layer3[-1].register_forward_hook(hook_t)

    self.criterion = torch.nn.MSELoss(reduction='sum')

    # PatchCore の別のメソッドで定義する
    self.init_results_list()

    self.data_transforms = transforms.Compose([
                        #transforms.Resize((load_size, load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        #transforms.CenterCrop(input_size),
                        transforms.RandomRotation(90),
                        transforms.ColorJitter(brightness=0.05,contrast=0.05),
                        transforms.Normalize(mean=mean_train, std=std_train)
                        ])
    self.test_transforms = transforms.Compose([
                        #transforms.Resize((load_size, load_size)),
                        transforms.ToTensor(),
                        #transforms.CenterCrop(input_size)
                        ])

    self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

  def init_results_list(self):
    self.test_result_annomaly_map = []
    self.input_img = []
    self.annomaly_score = []
    self.heatmap = []

  def init_features(self):
    self.features = []

  def forward(self, x_t):
    self.init_features()
    _ = self.model(x_t)
    return self.features

  def train_dataloader(self):
    train_datasets = MyDataset(img_set = self.hparams.gayoshi_train_img_list,
                               transform = self.data_transforms
                              )
    train_loader = DataLoader(train_datasets,
                              batch_size = self.hparams.batch_size,
                              shuffle = True,
                               num_workers = 0)
    return train_loader

  def test_dataloader(self):
    test_datasets = MyDataset(img_set=self.hparams.croped_img_list,
                              transform = self.test_transforms
                               )
    test_loader = DataLoader(test_datasets,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 0)
    return test_loader

  def configure_optimizers(self):
    return None

  def on_train_start(self):
    self.model.eval()
    self.embedding_dir_path = prep_dirs(self.hparams.project_root_path)#(self.logger.log_dir)
    self.embedding_list = []

  def on_test_start(self):
    self.index = faiss.read_index(os.path.join('pre_index.faiss'))#事前学習済みのファイルを使う
    #self.index = faiss.read_index(os.path.join('tmp', 'index.faiss'))
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
    self.init_results_list()

  # バッチごとに画像をモデルに通して抽出された特徴量が追加されていく
  def training_step(self, batch, batch_idx):
    x = batch
    features = self(x)
    embeddings = []
    for feature in features:
      m = torch.nn.AvgPool2d(3, 1, 1)
      embeddings.append(m(feature))
    embeddings = embedding_concat(embeddings[0], embeddings[1])
    self.embedding_list.extend(reshape_embedding(np.array(embeddings)))
    return None

  def test_step(self, batch, batch_idx):
    x  = batch
    features = self(x)
    embeddings = []
    for feature in features:
        m = torch.nn.AvgPool2d(3, 1, 1)
        embeddings.append(m(feature))
    embedding_ = embedding_concat(embeddings[0], embeddings[1])
    embedding_ = np.array(reshape_embedding(np.array(embedding_)))
    score_patches, _ = self.index.search(embedding_, k=self.hparams.n_neighbors)
    #print('デバッグ', score_patches.shape, score_patches)
    anomaly_map = score_patches[:,0].reshape((32,32))
    N_b = score_patches[np.argmax(score_patches[:,0])]#最も近い特徴同士の中で最大→画像の中の最も異常な部分のデータバンクからの距離
    w = (1 - (np.max(np.exp(N_b[0]))/np.sum(np.exp(N_b))))
    score = w*N_b[0] # Image-level score
    #画像表示の際に使うためにN_bの値が最大のものを保存しておく
    if N_b[0] > self.max_Nb:
      self.max_Nb = N_b[0]
    anomaly_map_resized = cv2.resize(anomaly_map, (self.hparams.input_size, self.hparams.input_size))
    anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
    x = self.inv_normalize(x)
    input_img = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
    self.test_result_annomaly_map.append(anomaly_map_resized_blur)
    self.input_img.append(input_img)
    self.annomaly_score.append(score)