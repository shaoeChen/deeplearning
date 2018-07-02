# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:06:02 2018

@remark:
    實作模型中的實用小工具:
        1. 照片轉矩陣:image_to_matrix
        2. 資料分割並做亂數排序:shuffle_index
        3. 照片檢閱:plt_image
@author: marty.chen
"""

from skimage import io, color
import numpy as np
import math

def image_to_matrix(path_folder, path_file, image_exten='jpg', as_gray=False, label_num=None, gray2img=False):
    """
    將資料夾路徑內的所有照片(依image_exten設置)轉為numpy array
    如果需要同步產生等量的label向量可利用label_num來設置    
    環境內必需有安裝skimage，若無法滿足則需另外以pillow來執行，安裝anaconda的時候也會擁有skimage
    
    parameter:
        path_folder:資料夾路徑
        path_file:檔案清單，格式為list，可利用os.listdir來取得資料夾內的檔案清單
        image_exten:照片副檔名，預設為jpg
        as_gray:是否灰值化
        label:如果要產生label清單的話就輸入label，會依path_file長度來產生相對長度的類別清單
        gray2img:將灰度圖轉rgb
    
    return:
        datasets:numpy array(m, n_h, n_w, n_c)
            如果轉了灰度圖就不會有n_c
        label:numpy array，該資料夾的label(m, )
        
    example:
        取得一個dataset，副檔名為bmp，label為1
        ds_ng_1, ds_label_1 = image_to_matrix(path_ng_1, file_ng_1, 'bmp', False, 1, True)        
    """
    _datasets = []
    for file in path_file:       
        if file.endswith(image_exten):
            if gray2img:
                img_gray = io.imread(path_folder + file, as_grey=as_gray)
                img = color.gray2rgb(img_gray)
                _datasets.append(img)
            else:
                _datasets.append(io.imread(path_folder + file, as_grey=as_gray))
        continue
            
    datasets = np.array(_datasets).astype('float32')    
        
    
    if isinstance(label_num, int):
        #  必需為int格式
        label = np.ones(len(_datasets)) * label_num      
    else:
        label = None
        
    return datasets, label


def shuffle_index(datasets, labels, train_size=0.8, random_seed=0):
    """
    將資料集做亂序排序，避免資料過於有序，並且切割為訓練資料集與測試資料集
    
    parameter:
        datasets:原始資料集，格式為(m, other)
            m:資料集數量
            other:為其它維度
            預設np.random.permutation會以datasets.shape[0]來設置索引長度
        lables:目標類別        
        train_size:訓練集比例
            train = train-size * m
            test = m - train
        random_seed:亂數種子，預設為0，設置完善的話可以確保每次執行的結果相同
    
    return:
        X_train, X_test, y_train, y_test:四個分割後的資料集
        (m, train_m, test_m): 總數量、訓練集數量、測試驗證集數量
        
    不管你願不願意，執行這個function都會把你的資料集重新洗牌
    後續的label轉one_hot可利用keras.np_utils.to_categorical來完成
    """
    np.random.seed(random_seed)
    m = datasets.shape[0]
    
    shuffle_index = np.random.permutation(m)
    
    train_m = int(m * train_size)
    test_m = m - train_m
    
    #  重新賦值
    datasets = datasets[shuffle_index]
    labels = labels[shuffle_index]

    #  資料分割
    X_train = datasets[:train_m]
    X_test = datasets[train_m:]
    y_train = labels[:train_m]
    y_test = labels[train_m:]
    
    #  資料集驗證
    assert (X_train.shape[0] == train_m)
    assert (X_test.shape[0] == test_m)
    assert (y_train.shape[0] == train_m)
    assert (y_test.shape[0] == test_m)
    assert (m == train_m + test_m)
    
    return X_train, X_test, y_train, y_test, (m, train_m, test_m)


def plt_image(images, labels, predict_labels=[], idx_start=0, idx_batch_size=10, cel_num=5, fig_size='big'):
    """
    用來檢閱照片，了解各照片的原始label與預測的label，以便於理解誤判原因
    
    parameter:
        images:來源照片(x)
        labels：來源照片類別(y)
        predict_labels:預測照片類別
        idx_start:起始索引
            預設為0
        idx_batch_size:每次讀取量
            預設為10
        cel_num:每row顯示幾張照片
            預設為5
        fig_size:figure.figsize設置
            big:16,9
            middle:12,7
            small:8,4
        
    當每次讀取批量>20的時候會以20取值
    如果照片想要大點來看，就必需將cel_num設置小一點，然後設置size為big或是middle
    """
    #  判斷索引值是否超過10，若超過20則idx_batch_size重新賦值
    if idx_batch_size > 20:
        idx_batch_size = 20   
    
    rows = int(math.ceil(idx_batch_size / cel_num))
    
    if fig_size=='big':
        _size=(16, 9)
    elif fig_size=='middle':
        _size=(12, 7)
    elif fig_size=='small':
        _size=(8, 4)
        
    #  設置圖表尺寸
    plt.figure(figsize=(_size))
    for i in range(idx_batch_size): 
        ax = plt.subplot(rows,cel_num,i+1)
        #  取消x、y軸的刻度
        plt.xticks(())        
        plt.yticks(())
        #  設置x軸的label為
        ax.set_xlabel('True labels:' + str(labels[idx_start]) + ',idx:' + str(idx_start))
        #  如果predict_labls不是空值，那就帶入資料
        if len(predict_labels)>0:            
            ax.set_title('Predict labels:' + str(predict_labels[idx_start]))
        ax.imshow(images[idx_start])
        #  換下一筆
        idx_start += 1        
         
    #  確保資料呈現正常    
    plt.tight_layout()    
    plt.show()        