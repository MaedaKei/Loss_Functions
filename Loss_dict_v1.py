import torch
import torch.nn as nn

"""
ユーザー定義の損失関数をここに記述し、Loss_dictに登録する
"""
class FocalLoss(nn.Module):
    def __init__(
            self,
            gamma=0,
            class_weights=1#引数指定するときはリスト型
    ):
        super(FocalLoss,self).__init__()
        self.gamma=gamma
        self.class_weights=class_weights
        self.smooth=1e-7
    def __call__(
            self,
            pred,#予測した確率分布　(B,C,H,W) or (B,C,x,y,z) tanhにより-1~1
            target#真の確率分布 (B,C,H,W) or (B,C,x,y,z) 
    ):
        #(C,N)=(C,BHW) or (C,BDHW)
        #Pは画像の各位置にone-hotベクトルが立っているイメージ
        #Qは画像の各位置にクラス数分の確率が立っているイメージ
        #P,Qともに(C,*)の形に変形する。
        pred_flat=torch.clamp(pred.transpose(1,0).reshape(pred.shape[1],-1),min=self.smooth,max=1-self.smooth)
        target_flat=torch.clamp(target.transpose(1,0).reshape(target.shape[1],-1),min=self.smooth,max=1-self.smooth)
        
        #(C,N) -> (1,N)
        #log(predict)の部分に0がまぎれると-infが発生する。そこに０をかけるとnanが発生するのでself.smoothを加算したものの対数を取って０対数を回避する
        res=self.class_weights*((1-pred_flat)**self.gamma)*torch.log(pred_flat)*target_flat#(C,*)
        FL=-1*torch.mean(res)#(C,*) -> (1,1)
        return FL
class DiceLoss(nn.Module):
    def __init__(
            self,smooth=1e-7,class_weight=1.0
    ):
        self.smooth=smooth
        self.class_weight=class_weight
    def __call__(self,pred,target):
        pred_flat=torch.clamp(pred.transpose(1,0).reshape(pred.shape[1],-1),min=self.smooth,max=1-self.smooth)
        target_flat=torch.clamp(target.transpose(1,0).reshape(target.shape[1],-1),min=self.smooth,max=1-self.smooth)
        res=torch.sum(pred_flat*target_flat,dim=1)*self.class_weight#(C,1)
        C_loss=1-res/torch.sum(pred_flat*pred_flat+target_flat*target_flat,dim=1)
        #loss=torch.mean(self.class_weight*C_loss)#(C,1)->(1,1)
        loss=torch.mean(self.class_weight*C_loss)
        return loss
class FocalTverskyLoss(nn.Module):
    def __init__(
            self,
            FN=0.7,FP=0.3,FTS_gamma=4/3,#1/gammaでつかうやつ
            class_weights=1,smooth=1e-6
    ):
        super(FocalTverskyLoss,self).__init__()
        self.FN=FN
        self.FP=FP
        self.gamma=FTS_gamma
        self.class_weights=class_weights
        self.smooth=smooth
    def __call__(self,pred,target):
        # 0 <= (pred target) <=1
        #(B,C,H,W) or (B,C,D,H,W) -> (C,*)
        pred_flat=torch.clamp(pred,min=self.smooth,max=1.0).transpose(1,0).reshape(pred.shape[1],-1)
        target_flat=torch.clamp(target,min=self.smooth,max=1.0).transpose(1,0).reshape(target.shape[1],-1)
        TP=torch.sum(target_flat*pred_flat,dim=1)#(C,1)
        FN=torch.sum(target_flat*(1.0-pred_flat),dim=1)#(C,1)
        FP=torch.sum((1.0-target_flat)*pred_flat,dim=1)#(C,1)
        Tversky_loss=1.0-1.0/(1.0+(self.FN*FN+self.FP*FP)/TP)#(C,1)
        #FocalTversky=torch.mean(self.class_weights*(Tversky_loss**(1/self.gamma)))
        FocalTversky=torch.mean(self.class_weights*(Tversky_loss**(1/self.gamma)))
        print(f"TP={TP}\nFN={FN}\nFP={FP}")
        print(f"Tversky_loss={Tversky_loss}")
        print(f"FocalTversky_loss={FocalTversky}\n")
        return FocalTversky
#loss functionはnnの下のものを使用している
#研究タスクに向いている損失関数に適宜書き換える必要がある
#辞書のkeyに設定している文字列を数式中に記述する。
#O: 4*L1+L2
#X: 4*L1(X,y)+L2(X,y)
#X: 4*nn.L1Loss+nn.MSELoss
Loss_dict={
    "L1":nn.L1Loss(),"L2":nn.MSELoss(),
    ##CrossEntropy系統
    "CE":FocalLoss(gamma=0),"FL":FocalLoss(gamma=2),
    ##Dice系統
    #"DICE":FocalTverskyLoss(FN=0.5,FP=0.5,FTS_gamma=1),
    "DICE":DiceLoss(),
    "TS":FocalTverskyLoss(FN=0.7,FP=0.3,FTS_gamma=1),
    "FTS":FocalTverskyLoss(FN=0.7,FP=0.3,FTS_gamma=4/3),
    #Boundary系統
    "Boundary":None
}