# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:32:44 2018
@remark:
    keras的擴充實用小工具，需應用在有安裝keras的平台
    1. 預測概率轉標籤:utils_predict
    2. 產生圖面:model_history_plot
	3. utils_predict_validator
	4. plt_image_from_generator
@author: marty.chen
"""
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

def utils_predict(model, datasets, evaluate=True, need_prob=False):
    """
    keras預測之後回傳為概率，透過此function可轉為類別
    再依需求調控是否回傳概率與模型效能評估
    
    model:keras.model，訓練之後的模型
    datasets:資料集，格式為dict，索引名稱必需為X_train, X_test, y_train, y_test
        datasets:
            'X_train':train_dataset
            'X_test':test_dataset
            'y_train':train_label
            'y_test':test_label
    evaluate:是否執行model.evaluate，如果是，那資料集就必需有y_train與y_test
    need_prob:是否回傳概率
    
    return datasets
        預測類別：
            train_predict, test_predict
        預測概率：
            train_predict_prob, test_predict_prob
        模型評估：
            train_score, test_score
        
    datasets:dict    
    """
    _train_dataset = datasets['X_train']
    _test_dataset = datasets['X_test']
    
    _train_prob = model.predict(_train_dataset, verbose=1)
    _test_prob = model.predict(_test_dataset, verbose=1)
    
    _train_prob_labels = _train_prob.argmax(axis=-1)
    _test_prob_labels = _test_prob.argmax(axis=-1)
    
    datasets['train_predict'] = _train_prob_labels
    datasets['test_predict'] = _test_prob_labels
    
    if evaluate:
        _train_label = datasets['y_train']
        _test_label = datasets['y_test']
        scores_train = model.evaluate(_train_dataset, _train_label)
        scores_test = model.evaluate(_test_dataset, _test_label)
        datasets['train_score'] = scores_train[1]
        datasets['test_score'] = scores_test[1]
        
    if need_prob:
        datasets['train_predict_prob'] = _train_prob
        datasets['test_predict_prob'] = _test_prob
            
    return datasets


#  產生圖面(acc，val)
def model_history_plot(history, epoch):
    """
    繪製模型的訓練效能以及成本函數的收斂成果
    
    parameter:
        history:訓練記錄，為keras.History.history
        epoch:迭代次數，為keras.History.epoch
    
    首先判斷物件長度是否為4，如果是那代表有啟用驗證資料集，若為2，就代表只有訓練資料集。
    """
    #  用以判斷是否有驗證資料集，如果dict長度為4則賦值為True
    val = False
    
    if len(history)==4:
        val = True
        val_acc = history['val_acc']
        val_loss = history['val_loss']
        
    acc = history['acc']
    loss = history['loss']
       
    #  設置圖表
    plt.figure(figsize=(16,9))
    
    #  準確度_acc
    plt.subplot(121)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xlim(min(epoch), max(epoch)+1)
    plt.plot(epoch, acc, label='train_acc')
    
    if val:
        plt.plot(epoch, val_acc, label='val_cc')
    plt.legend(loc='best')
    
    #  訓練成本_loss
    plt.subplot(122)    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(min(epoch), max(epoch)+1)
    plt.plot(epoch, loss, label='train_loss')
    
    if val:
        plt.plot(epoch, val_loss, label='val_loss')            
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

	
def utils_predict_validator(model, datasets, need_prob=False):
    """
    keras預測之後回傳為概率，透過此function可轉為類別
    再依需求調控是否回傳概率與模型效能評估
    
    與utils_predict的不同在於，此function應用於新進資料驗證模型，並且一定執行predict與evaluate
    
    parameter:
        model:keras.model，訓練之後的模型
        datasets:資料集，格式為dict，索引名稱必需為dataset_vali(X)與label_vali(y)
            datasets:
                'dataset_vali':dataset
                'label_vali':label
        need_prob:是否回傳概率
    
    return:
        datasets
            預測類別：
                vali_predict
            預測概率：
                vali_predict_prob
            模型評估：
                vali_score
        
    datasets:dict    
    """
    #  取得資料集與標籤
    _dataset_vali = datasets['dataset_vali']
    _label_vali = datasets['label_vali']
    
    #  計算最大機率
    _label_prob = model.predict(_dataset_vali, verbose=1)
    
    #  取得最大機率值對應label
    _labels = _label_prob.argmax(axis=-1)
    
    #  資料寫入dict
    datasets['vali_predict'] = _labels 
    
    #  計算模型效能
    _model_scores = model.evaluate(_dataset_vali, _label_vali)

    #  資料寫入dict
    datasets['vali_score'] = _model_scores[1]    

    if need_prob:
        datasets['vali_predict_prob'] = _label_prob
            
    return datasets	
	
def plt_image_from_generator(generator, mask=[], predict_labels=[], idx_start=0, idx_batch_size=10, cel_num=5, fig_size='big'):
    """
    用來檢閱照片，了解各照片的原始label與預測的label，以便於理解誤判原因
    此function需搭配keras.generator使用，利用生成器取得照片
    
    parameter:
        generator: 來源照片生成器        
        mask: 切片遮罩
            如果想針對錯誤資料做檢閱，就可以先設置一個錯誤資料遮罩做為參數
        predict_labels: 預測照片類別
        idx_start: 起始索引
            預設為0
        idx_batch_size: 每次讀取量
            預設為10
        cel_num: 每row顯示幾張照片
            預設為5
        fig_size: figure.figsize設置
            big: 16,9
            middle: 12,7
            small: 8,4
        
    當每次讀取批量>20的時候會以20取值
    如果照片想要大點來看，就必需將cel_num設置小一點，然後設置size為big或是middle
    
    example:
        plt_image_from_generator(generator=train_generator, 
                         mask=mask_train,
                         predict_labels=train_predict)
    """
    #  判斷索引值是否超過20，若超過20則idx_batch_size重新賦值
    if idx_batch_size > 20:
        idx_batch_size = 20   
        
    #  0713_加入判斷，當資料筆數不足idx_batch_size，則以資料量為主
    if generator.samples < idx_batch_size:
        idx_batch_size = generator.samples
    
    rows = int(math.ceil(idx_batch_size / cel_num))
    
    if fig_size=='big':
        _size=(16, 9)
    elif fig_size=='middle':
        _size=(12, 7)
    elif fig_size=='small':
        _size=(8, 4)
    
    #  取得圖片索引資訊(檔案路徑)
    files = []
    
    if len(mask):
        indexs = generator.index_array[mask][idx_start: idx_start+idx_batch_size]
    else:
        indexs = generator.index_array[idx_start: idx_start+idx_batch_size]        
        
    for idx in indexs:
        file_path = os.path.join(generator.directory,generator.filenames[idx])
        true_label = generator.classes[idx]
        try:
            predict_labels = predict_labels[idx]
        except IndexError:
            predict_labels is None

        files.append(
            (
                file_path, 
                true_label, 
                predict_labels,
                idx
            )
        )

    #  繪製圖表
    plt.figure(figsize=(_size))
    i = 0
    for file in files:
        ax = plt.subplot(rows,cel_num,i+1)
        #  取消x、y軸的刻度
        plt.xticks(())        
        plt.yticks(())
        #  設置x軸的label為
        ax.set_xlabel('True labels:' + str(file[1]) + ',idx:' + str(file[3]))
        #  如果predict_labls不是空值，那就帶入資料
        if file[2]:
            ax.set_title('Predict labels:' + str(file[2]))        
        img = io.imread(file[0])
        ax.imshow(img)
        #  換下一筆
        i += 1                          
         
    #  確保資料呈現正常    
    plt.tight_layout()    
    plt.show()  	