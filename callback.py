from pytorch_lightning.callbacks import Callback
import os
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
import faiss
import cv2
from scipy.ndimage import gaussian_filter
from utils import min_max_norm, cvt2heatmap, heatmap_on_image

class MyCallback(Callback):
  def on_train_epoch_end(self, trainer, pl_module):
    total_embeddings = np.array(pl_module.embedding_list)

    #RandomProjection
    pl_module.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
    pl_module.randomprojector.fit(total_embeddings)

    #Coreset Subsampling
    selector = kCenterGreedy(total_embeddings, 0, 0)
    selected_idx = selector.select_batch(model=pl_module.randomprojector,
                                        already_selected=[],
                                        N=int(total_embeddings.shape[0]*0.001))
    pl_module.embedding_coreset = total_embeddings[selected_idx]

    print('initial embedding size : ', total_embeddings.shape)
    print('final embedding size : ', pl_module.embedding_coreset.shape)

    #faiss
    pl_module.index = faiss.IndexFlatL2(pl_module.embedding_coreset.shape[1])
    pl_module.index.add(pl_module.embedding_coreset)
    faiss.write_index(pl_module.index,
                      os.path.join(pl_module.embedding_dir_path, 'index.faiss'))
  
  #2024/03/23 テスト後にannomalyマップを作成する
  def on_test_epoch_end(self, trainer, pl_module):
    """
    for num, score in enumerate(pl_module.annomaly_score):
      anomaly_map = pl_module.test_result_annomaly_map[num]
      input_img = pl_module.input_img[num]
      if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
      anomaly_map_norm = min_max_norm(anomaly_map, pl_module.max_Nb)
      # anomaly map on image
      heatmap = cvt2heatmap(anomaly_map_norm*255)
      hm_on_img = heatmap_on_image(heatmap, input_img)
      pl_module.heatmap.append(hm_on_img)
    """
    
