#入力サイズ
Height =256 
Width = 256

#クラスラベル
Class_label = [
    'hitomi\\',
    'etc\\'
]

#クラス数
Class_num = len(Class_label)

#学習データのパス
Train_dirs = [
    'Data\\Train\\hitomi\\',
    'Data\\Train\\etc\\'
]

#テストデータのパス
Test_dirs = [
    'Data\\Test\\hitomi\\',
    'Data\\Test\\etc\\'
]

#ミニバッチ
Minibatch =32 

#データ拡張(data_loader.pyで使用)
Horizontal_flip = False
Vertical_flip = False