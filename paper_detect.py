from PIL import Image
import pyheif
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import random

class gayoshi_danmen():

  def __init__(self, folder, input_list):

    self.folder = folder
    #canny法というエッジ検出のパラメータの初期値
    self.canny_para_chigiri = (250,280)
    self.canny_para_hasami = (150,480)
    self.canny_para_both = (250,280)

    #画像全体を保存する
    chigiri_list = self.mk_pic_list(input_list[0])
    both_list = self.mk_pic_list(input_list[1])
    hasami_list = self.mk_pic_list(input_list[2])

    self.image_dict = {"chigiri":chigiri_list,
                  "both":both_list,
                  "hasami":hasami_list
    }

  def heic_png(self, image_path):
    #データの読み込み
    heif_file = pyheif.read(image_path)
    print("heic形式のデータ→",heif_file)
    #読み込んだファイルをデータに変換する
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride
    )
    print("画像処理ライブラリPILの形式→",data)
    return data

  def mk_pic_list(self, input_list):
    pic_list = []
    for pic_name in input_list:
      pic_adress = self.folder + pic_name
      pic = self.heic_png(pic_adress)
      pic = np.array(pic)
      pic_list.append(pic)
    return pic_list

  def choice_pic(self, image_type, num):
    return self.image_dict[image_type][num]

  #canny法によるエッジ検出
  #canny_para=(230,270)が最初に試したときはいい感じだったので初期値としている
  def canny_edge_check(self,image_type, num, canny_para, range=20):

    original = self.choice_pic(image_type, num)
    th_1 = canny_para[0]
    th_2 = canny_para[1]

    img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    r = range
    check_canny_list = [[th_1-r,th_2+r],
                        [th_1  ,th_2+r],
                        [th_1+r,th_2+r],
                        [th_1-r,th_2  ],
                        [th_1  ,th_2  ],
                        [th_1+r,th_2  ],
                        [th_1-r,th_2-r],
                        [th_1,  th_2-r],
                        [th_1+r,th_2-r]
                        ]

    edge_pic_list = []
    canny_para_list = []

    for i in check_canny_list:
      edge_pic_temp = cv2.Canny(img_gray, i[0], i[1])
      edge_pic_list.append(edge_pic_temp)
      canny_para_list.append(i)

    self.show_pics(edge_pic_list, 3, 3, 0, "x→threshold1, y→threshold2", (24,32), canny_para_list)

    #表示した画像でのcannyパラメータを代入する
    if image_type == "chigiri":
      self.canny_para_chigiri = canny_para
    elif image_type == "hasami":
      self.canny_para_hasami = canny_para
    else:
      self.canny_para_both = canny_para
      
  #canny法の時に画像を表示させるための関数
  def show_pics(self, data, row ,col, start_num, title, fig_size, canny_para=(0,0)):

    fig, ax = plt.subplots(nrows=row, ncols=col,figsize=fig_size)

    num_data = row*col
    fig.suptitle(title , fontsize=24, color='black')
    fig.subplots_adjust(hspace=0.15, wspace=0.01)

    #skip_data = [data[(i+1)*200] for i in range(num_data)]

    for i, img in enumerate(data[start_num:start_num+num_data]):

      _r= i//col
      _c= i%col
      #ax[_r,_c].set_title(skip_data[i], fontsize=16, color='white')
      ax[_r,_c].axes.xaxis.set_visible(False) # X軸を非表示に
      ax[_r,_c].axes.yaxis.set_visible(False) # Y軸を非表示に
      ax[_r,_c].imshow(img, cmap="gray") # 画像を表示
      # タイトルとしてパラメータを表示（画像の上部）
      if not canny_para==(0,0):
        title_text = f"Canny Params: {canny_para[i]}"
        ax[_r, _c].set_title(title_text, fontsize=30, color='green', pad=3)

  def search_danmen(self, size):

    danmen_dict_original = {}
    danmen_dict_gray = {}

    for im_type in ["chigiri","both","hasami"]:

      danmen_list_gray = []
      danmen_list_original = []
      count = 0

      for img in self.image_dict[im_type]:

        half_size = (int(size[0]/2), int(size[1]/2))

        for k in [(0,0), half_size]:

          img = img[k[0]:,k[1]:]

          if im_type == "chigiri":
            canny_para = self.canny_para_chigiri
          elif im_type == "hasami":
            canny_para = self.canny_para_hasami
          else:
            canny_para = self.canny_para_both

          img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          img_edge = cv2.Canny(img_gray, canny_para[0], canny_para[1])

          col_num = int(img.shape[0]/size[0])
          row_num = int(img.shape[1]/size[1])

          for col in range(col_num-1):
            for row in range(row_num-1):

              img_part_edge = img_edge[col*size[0]:(col+1)*size[0],
                                      row*size[1]:(row+1)*size[1]]
              img_part_gray = img_gray[col*size[0]:(col+1)*size[0],
                                      row*size[1]:(row+1)*size[1]]
              img_part = img[col*size[0]:(col+1)*size[0],
                            row*size[1]:(row+1)*size[1]]

              # 画像の端5ピクセル内に255が存在するかどうかをチェック
              top_edge = img_part_edge[:5, :]  # 上端
              bottom_edge = img_part_edge[-5:, :]  # 下端
              left_edge = img_part_edge[:, :5]  # 左端
              right_edge = img_part_edge[:, -5:]  # 右端

              # 上下左右の端のいずれかに255が含まれているか
              if (255 in top_edge) and (255 in bottom_edge) or (255 in left_edge) and (255 in right_edge):
                danmen_list_original.append(img_part)
                danmen_list_gray.append(img_part_gray)

        count += 1
        print(im_type, count, "→完了")

      danmen_dict_original[im_type] = danmen_list_original
      danmen_dict_gray[im_type] = danmen_list_gray

      print(im_type, "の数→", len(danmen_list_original))

    self.danmen_dict = [danmen_dict_original, danmen_dict_gray]

    return danmen_dict_original, danmen_dict_gray

  #画像のリストをランダムに分ける関数を作成

  def train_sampling(self, img_type, rate, number=1000):

    pic_list = self.danmen_dict[1][img_type]

    #ランダムに並び替えて、スライスで必要な分を取り出す。
    random_sample = random.sample(pic_list, min(number, len(pic_list)))
    #→ランダムにサンプルを並べ直すと、隣同士の画像がtrain,valにそれぞれ入って、カンニングみたいな状態になる
    train_data = pic_list[:int(len(pic_list)*rate)]
    test_data = pic_list[int(len(pic_list)*rate):]

    return train_data, test_data