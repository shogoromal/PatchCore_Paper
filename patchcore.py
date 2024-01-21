
import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import faiss
from utils import min_max_norm, cvt2heatmap, embedding_concat, reshape_embedding, prep_dirs, heatmap_on_image
from dataset import MyDataset

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class PatchCore(pl.LightningModule):

  def __init__(self, hparams):

    super(PatchCore, self).__init__()
    self.save_hyperparameters()

    # PatchCore の別のメソッドで定義する
    self.init_features()

    #何個目のテスト画像かを数えるための変数
    self.testcount = 0

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
    self.gt_transforms = transforms.Compose([
                        #transforms.Resize((load_size, load_size)),
                        transforms.ToTensor(),
                        #transforms.CenterCrop(input_size)
                        ])

    self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

  def init_results_list(self):
    self.gt_list_px_lvl = []
    self.pred_list_px_lvl = []
    self.gt_list_img_lvl = []
    self.pred_list_img_lvl = []
    self.img_path_list = []

  def init_features(self):
    self.features = []

  def forward(self, x_t):
    self.init_features()
    _ = self.model(x_t)
    return self.features

  def save_anomaly_map(self, anomaly_map, input_img, file_name, x_type):
    if anomaly_map.shape != input_img.shape:
      anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
    anomaly_map_norm = min_max_norm(anomaly_map)
    anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm*255)
    hm_on_img = heatmap_on_image(heatmap, input_img)

    # save images
    cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
    cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
    cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
    #cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)


  def train_dataloader(self):
    image_datasets = MyDataset(img_set = self.boundary_dict_original["hasami"],
                               transform = self.data_transforms
                              )
    train_loader = DataLoader(image_datasets,
                              batch_size = self.batch_size,
                              shuffle = True,
                               num_workers = 0)
    return train_loader

  def test_dataloader(self):
    test_datasets = MyDataset(img_set = self.boundary_dict_original["both"],
                              transform = self.gt_transforms
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
    self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.project_root_path)#(self.logger.log_dir)
    self.embedding_list = []

  def on_test_start(self):
    self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.project_root_path)
    self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
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

  def test_step(self, batch, batch_idx): # Nearest Neighbour Search
    #filenameが一意になるように工夫が必要
    x  = batch
    label = "chigiri"
    x_type = "x_type"
    # extract embedding
    features = self(x)
    embeddings = []
    for feature in features:
        m = torch.nn.AvgPool2d(3, 1, 1)
        embeddings.append(m(feature))
    embedding_ = embedding_concat(embeddings[0], embeddings[1])
    embedding_test = np.array(reshape_embedding(np.array(embedding_)))
    score_patches, _ = self.index.search(embedding_test , k=self.n_neighbors)
    anomaly_map = score_patches[:,0].reshape((32,32))
    N_b = score_patches[np.argmax(score_patches[:,0])]
    w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
    score = w*max(score_patches[:,0]) # Image-level score
    #gt_np = gt.cpu().numpy()[0,0].astype(int)
    anomaly_map_resized = cv2.resize(anomaly_map, (self.input_size, self.input_size))
    anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
    #self.gt_list_px_lvl.extend(gt_np.ravel())
    self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
    #self.gt_list_img_lvl.append(label.cpu().numpy()[0])
    self.pred_list_img_lvl.append(score)
    #self.img_path_list.extend(self.testcount)
    # save images
    x = self.inv_normalize(x)
    input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
    self.save_anomaly_map(anomaly_map_resized_blur, input_x, self.testcount, x_type)
    self.testcount += 1