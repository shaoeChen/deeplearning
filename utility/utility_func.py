# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:06:02 2018

@remark:
    實作模型中的實用小工具:
        1. 照片轉矩陣:image_to_matrix
        2. 資料分割並做亂數排序:shuffle_index
        3. 照片檢閱:plt_image
		4. 照片維度檢核:dimension_check
        5. 資料分割小批量:random_batch
@module:
        1. scikit-image==0.14
@author: marty.chen
"""

from skimage import io, color
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import glob
import shutil
import random


def image_to_matrix(path_file, path_folder=None, image_exten='jpg', as_gray=False, label_num=None,
                    gray2img=False, path_file_only=False, img_resize=None):
    """
    將資料夾路徑內的所有照片(依image_exten設置)轉為numpy array
    如果需要同步產生等量的label向量可利用label_num來設置
    環境內必需有安裝skimage，若無法滿足則需另外以pillow來執行，安裝anaconda的時候也會擁有skimage

    parameter:
        path_file ->str: 檔案清單，格式為list，可利用os.listdir來取得資料夾內的檔案清單
        path_folder ->str: 資料夾路徑，當path_file_only為False的時候不得不None
        image_exten ->str: 照片副檔名，預設為jpg
        as_gray ->bool: 是否灰值化
        label ->int:如果要產生label清單的話就輸入label，會依path_file長度來產生相對長度的類別清單
        gray2img ->bool: 將灰度圖轉rgb
        path_file_only ->bool: 部份情況下可能可以直接提供整個檔案清單，設置為True可不設置path_folder
        img_resize ->tuple: 縮放之後的大小
            resize之後，資料格式將從`np.uint8`變更為`np.float64`
            因此實作上有乘上255並回轉為`np.uint8`
            見參考來源說明

    return:
        datasets: numpy array(m, n_h, n_w, n_c)
            如果`gray2img=True`不會有n_c，因此使用上記得`reshape`或加入一個軸(axis)
        label: numpy array，該資料夾的label(m, )

    resource:
        參考來源_resize：https://stackoverflow.com/questions/44257947/skimage-weird-results-of-resize-function
    example:
        1. 使用path_file搭配path_folder
            取得一個dataset，副檔名為bmp，label為1
            ds_ng_1, ds_label_1 = image_to_matrix(file_ng_1, path_ng_1, 'bmp', False, 1)
        2. 也可以試著利用glob來取得檔案清單
            full_path = glob.glob('d:\abc\*.jpg')
            這樣就可以取得abc資料夾底下所有的jpg的完整路徑清單
            ds_ng1, ds_label_1 = image_to_matrix(full_path, None, jpg, False, 1, False, True)
    """
    #  如果不是單純的上傳檔案路徑清單則path_folder一定要有東西，不能為None
    #  若為None就直接回傳None, None避免程式中斷
    if path_file_only == False and path_folder is None:
        return None, None

    _datasets = []
    for file in path_file:
        #  確認副檔名正確才處理，不正確的直接pass掉
        if file.endswith(image_exten):
            #  依不同資料來源格式分別讀入照片
            if path_file_only:
                img = io.imread(file, as_gray=as_gray)
            else:
                img = io.imread(os.path.join(path_folder, file), as_gray=as_gray)

                #  是否將灰度圖轉rgb照片
            if gray2img:
                img = color.gray2rgb(img_gray)

            #  是否縮放照片
            if img_resize:
                img = resize(img, img_resize)
                img = img * 255
                img = img.astype(np.uint8)

            _datasets.append(img)
        else:
            continue

    datasets = np.asarray(_datasets)

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
    #  判斷索引值是否超過20，若超過20則idx_batch_size重新賦值
    if idx_batch_size > 20:
        idx_batch_size = 20   
        
    #  0713_加入判斷，當資料筆數已不足idx_batch_size，則以資料量為主
    if images.shape[0] < idx_batch_size:
        idx_batch_size = images.shape[0]
    
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
	



def dimension_check(vali_shape, path_file, image_exten='jpg', path_folder=None, path_file_only=False):
    """
    description:
        照片維度驗證，不少情況下資料集內總是會躲著大小與需求維度不同的照片，透過簡單的function做一個快速的檢核 
        將維度與需求不符的照片列出，方便排除
        
    parameter:
        vali_shape:tuple，驗證的dimension(n_H, n_W, n_C)
        path_file:檔案清單，格式為list，可利用os.listdir來取得資料夾內的檔案清單，或利用glob取得清單
        path_folder:資料夾路徑，當path_file_only為False的時候不得不None
        image_exten:照片副檔名，預設為jpg
        path_file_only:部份情況下可能可以直接提供整個檔案清單，設置為True可不設置path_folder
        
    return:
        沒有回傳
        
    example:
        利用glob取得清單:
            full_path = glob.glob('d:\abc\*.jpg')
            vali_shape=(155, 298, 3)
            dimension_check(vali_shape, full_path, 'jpg', None, True)
    """
    if path_file_only == False:
        if path_folder is None:
            #  利用return來中斷後面程序
            return None
        
    for file in path_file:       
        if file.endswith(image_exten):
            if path_file_only:        
                if io.imread(file).shape != vali_shape:
                    print('dimension_error:%s' % file)
            else:                
                if io.imread(os.path.join(path_folder,file)).shape != vali_shape:
                    print('dimension_error:%s' % file)
            
def random_batch(datasets, label, batch_size=64, seed=0):
    """
    describe:  
        在實作mini_batch的時候記得資料集跟label要一起當參數，因為兩者之間的關聯還是必需保存
        需注意資料格式為(特徵n, 資料集m)
        
    parameter:
        datasets:資料集(X)，dimension需為(n, m)
        label:標籤(y)，dimension需為(類別數, m)
        batch_size:每次訓練批量
        seed:亂數種子
        
    return:
        batch:[(X, y)]，格式為list，裡面的各小批量資料集為tuple
        
    example:
        dataset:(784, 50000)
        leabel:(10, 50000)
        
        batches = random_batch(dataset, label, batch_size=64, seed=10)
        for batch in batches:
            batch_X, batch_y = batch
            .....
    """
    #  先取得資料集總數，習慣上使用m，這是因為學習來自andrew的課程
    m = datasets.shape[0]
    batches = []
    np.random.seed(seed)
    
    #  利用np.random.permutation來取得資料集的亂數索引之後再重新調整資料索引
    #  這樣子資料集與label就有相同的索引排序了
    shuffle_index = np.random.permutation(m)
    shuffle_X = datasets[:, shuffle_index]
    shuffle_y = label[:, shuffle_index]
    
    #  計算需調整為幾個batch
    batch_num = math.floor(m / batch_size)
    for i in range(batch_num):
        #  每次取一個batch_size，代表每次取batch_size到batch_size+batch_size的切片
        batch_X = shuffle_X[:, batch_size * i: batch_size * i + batch_size]
        batch_y = shuffle_y[:, batch_size * i: batch_size * i + batch_size]
        batches.append((batch_X, batch_y))
        
    #  餘數沒有辦法處理的
    if m % batch_size != 0:
        batch_X = shuffle_X[:, batch_size * batch_num: m]
        batch_y = shuffle_y[:, batch_size * batch_num: m]
        batches.append((batch_X, batch_y))
        
    return batches
	


def move_file(source_data_path, subfolder_path, move_number, move_to_folder='test', random_seed=10):
    """
    description:
        模型訓練的時候如果有需求區分資料集為訓練資料集與測試資料集，讓keras可以至指定資料集直接讀取資料的時候
        可以利用這個function，指定資料集以及切割數量，快速將實體資料集分成兩個資料夾
        只針對jpg結尾檔案有效果，如有其餘需求再自行調整即可
        當move_number比該資料夾內檔案清單來的少的時候，會單純搬運該資料夾內數量
        
    parameter:
        source_data_path: 完整資料集來源資料夾
        subfolder_path: 完整資料集來源資料夾內的分類資料集
            實務上我們在區分資料的時候可能會分成貓、狗、豬並分三個資料夾存放，這邊即設置貓、狗、豬
        move_number: 搬移數量
            會將相對應的數量搬至指定資料夾
        move_to_folder: 預設將分割資料搬至source_data_path目錄下的test資料夾
            保存於test\subfolder_path\，避免搬移之後忘記自己搬去那了
        random_seed: 亂數種子
        
    example:
        資料結構如下：
            d:\data_source\
                貓\
                狗\
                豬\
        source_data_path='d:\data_source'
        subfolder_path='狗'
        move_to_folder='test'
            搬移之後檔案會轉過去d:\data_source\test\狗
	    
		
    """
    _exten = 'jpg'
    
    #  預計搬移的資料夾路徑組合成絕對路徑
    move_to_folder = os.path.join(source_data_path, move_to_folder)
    
    #  判斷資料夾是否存在
    if os.path.exists(move_to_folder):
        #  當資料夾存在的時候還需要進一步確認該資料夾下是否有subfolder_path
        if not os.path.exists(os.path.join(move_to_folder, subfolder_path)): 
            os.mkdir(os.path.join(move_to_folder, subfolder_path))
    else:
        #  如果不存在就代表連subfolder_path也不存在，直接一併建立
        os.makedirs(os.path.join(move_to_folder, subfolder_path))
        
    
    #  取得該子資料集路徑內的所有資料清單
    #  沒特別需求情況下預設為jpg
    data_list = glob.glob(os.path.join(source_data_path, subfolder_path) + '\*.' + _exten)
    #  取得路徑檔案總數
    data_number = len(data_list)
    #  當分割數比檔案還多的時候就讓分割數等同數量即可，不拋出異常
    if data_number < move_number:
        move_number = data_number
        
    #  亂數取得要搬移的檔案索引
    random.seed(random_seed)
    random_index = random.sample(range(data_number), move_number)

    #  檔案搬運
    for index in random_index:
        shutil.move(data_list[index], os.path.join(move_to_folder, subfolder_path))	