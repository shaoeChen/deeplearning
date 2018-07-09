# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:32:44 2018
@remark:
    keras的擴充實用小工具，需應用在有安裝keras的平台
    1. 預測概率轉標籤:utils_predict
    2. 產生圖面:model_history_plot
@author: marty.chen
"""
import matplotlib.pyplot as plt

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