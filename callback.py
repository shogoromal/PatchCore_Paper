from pytorch_lightning.callbacks import Callback
import os
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
import faiss

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
  
  def on_test_epoch_end(self, trainer, pl_module):
    None