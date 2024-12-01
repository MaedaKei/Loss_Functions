import torch
import torch.nn as nn
import torch.nn.functional as F
import re
#使用する損失関数をまとめたところ
from Loss_dict_v1 import Loss_dict

class OPERATOR:#演算子について
    def __init__(self,priority,functions,need_term):
        #演算子の優先度
        self.priority=priority
        #演算子の計算
        self.functions=functions
        #この演算子の計算に必要な項数
        self.need_term=need_term
    
    def __call__(self,X):
        #Xは必要数の項をまとめたリストになっている
        return self.functions(*X)

class Loss_Functions:
    """
    数式構文解析をして損失を計算する
    全体の損失関数、それぞれの損失関数のログも保存する
    """
    operators={
        "**":OPERATOR(2,lambda a,b:a**b,2),
        "*":OPERATOR(1,lambda a,b:a*b,2),
        "/":OPERATOR(1,lambda a,b:a/b,2),
        "+":OPERATOR(0,lambda a,b:a+b,2),
        "-":OPERATOR(0,lambda a,b:a-b,2)
    }
    operator_list=[i for i in operators.keys()]
    FULL_LOSS_NAME="ALL"
    def __init__(self,loss_text):
        dammy=loss_text
        dammy=re.sub("([\(\)])"," \\1 ",dammy)
        dammy=re.sub("([\+\-\*\/]+)"," \\1 ",dammy)
        self.token_list=[i for i in re.split(" +",dammy) if i!=""]
        self.temporary_log={Loss_Functions.FULL_LOSS_NAME:None}#損失関数ごとの値を保持するためのもの
        self.loss_name_set=set()
        #self.token_listは関数名か、係数か演算子で構成されるはずである。
        #5*L1+4*softmax*L2->["5","*","L1","+","4","*","softmax","*","L2"]
        stack=[]
        new_token_list=[]
        for token in self.token_list:
            """
            print(f"{token=}")
            print(new_token_list)
            print(stack)
            print('---------------------')
            """
            token_check=self.token_check(token)
            #print(token_check)
            if token_check=="numelic":
                new_token_list.append(token)
            elif token_check=="loss":
                new_token_list.append(token)
                #ログの辞書に
                self.temporary_log[token]=None
                self.loss_name_set.add(token)
            elif token_check=="operator":
                if len(stack)>0:
                    #スタックにたまってる演算子のなかで、自分以上の優先度のものを吐き出す
                    #スタックに演算子を追加する
                    stacked_op=stack[-1]
                    while self.token_check(stacked_op)=="operator" and Loss_Functions.operators[token].priority<=Loss_Functions.operators[stacked_op].priority:
                        new_token_list.append(stack.pop())
                        if len(stack)==0:
                            break
                        stacked_op=stack[-1]
                #現在の演算子をスタックに追加
                stack.append(token)
            
            #括弧の挙動
            elif token=="(":
                stack.append(token)
            elif token==")":
                #スタックの中に左括弧が現れるまで吐き出し続ける
                #右括弧が現れるということはstackに何かしら入っているはずなので確認省略
                stacked_op=stack.pop()
                while stacked_op!="(":
                    new_token_list.append(stacked_op)
                    stacked_op=stack.pop()
            else:
                print(f"よくわからないものが入っている: {token}")
                exit()
        
        #stackの中の余った演算子をtoken_listに追加していく
        new_token_list.extend(reversed(stack))
        self.token_list=new_token_list
        #数式解析結果をもとに損失関数を生成する
        workspace=[]
        for token in self.token_list:
            token_check=self.token_check(token)
            if token_check=="numelic":
                #定数も無理やり関数として扱うことで簡単な処理に帰着させる
                workspace.append(self.numelic_dec(token))
            elif token_check=="loss":
                workspace.append(self.loss_dec(token))
            elif token_check=="operator":#デコレートする
                operator=Loss_Functions.operators[token]
                #逆ポーランド記法から計算する際、スタックから取り出したトークンは取り出した時と逆順で演算子に渡す
                #inputリストを逆順にする必要があるが、reversed()を使うとイイテレートオブジェクトになって繰り返し使えなくなる
                #よって[::-1]とすることで順番を逆にする
                input=[workspace.pop() for _ in range(operator.need_term)][::-1]
                workspace.append(self.operator_dec(input,operator))
        self.loss_function=workspace[0]

    ##基礎関数デコレート
    #定数項部分
    def numelic_dec(self,num:str):
        return lambda:float(num)
    #定数項部分 (損失値)
    def loss_dec(self,loss:str):
        #self.temporary_logにはあらかじめ計算された損失値がある。
        return lambda:self.temporary_log[loss]
    #演算部分
    def operator_dec(self,func_list:list,operator):
        #function_listには関数が入っている
        return lambda:operator([func() for func in func_list])
    
    #損失計算
    def __call__(self,X=None,Y=None):
        #temporary_logの初期化
        """
        現時点ではself.temporary_logを内包表記で初期化したときにアドレスが変わるのか、
        そうなった場合,損失関数の参照場所がどうなるのかわからないので普通のfor文内包表記で初期化する
        """
        self.temporary_log={loss_name:Loss_dict[loss_name](X,Y) for loss_name in self.loss_name_set}
        """
        for loss_name in self.loss_name_set:
            self.temporary_log[loss_name]=Loss_dict[loss_name](X,Y)
        """
        loss=self.loss_function()#Tensor
        self.temporary_log[Loss_Functions.FULL_LOSS_NAME]=loss
        return loss
    
    def token_check(self,token):
        #演算子,数値,関数に分類する
        if token in Loss_Functions.operator_list:
            return "operator"
        try:
            #損失関数のキーワードか、係数かわからないのでfloatがたにキャストしてみる。
            float(token)
        except ValueError:
            #tokenが数字とアルファベットで構成されている→損失関数のキーワード
            if token.isalnum():
                return "loss"
        else:
            return "numelic"
        
        return "Error"

if __name__=="__main__":
    import numpy as np
    text="DICE"
    #x=torch.ones((5,5),requires_grad=True)
    #y=torch.zeros((5,5),requires_grad=True)
    loss_func=Loss_Functions(text)
    print(f"損失関数\n{text}")
    print(f"逆ポーランド\n{loss_func.token_list}\n-------------------------------")
    B,C,H,W=1,2,3,3
    EPOCH=1
    np.random.seed(0)
    for e in range(EPOCH):
        X=np.exp(np.random.rand(B,C,H,W))
        exp_sum=X.sum(axis=1,keepdims=True)
        X=X/exp_sum
        Y=np.argmax(X,axis=1)
        Y=np.eye(C)[Y].transpose(0,3,1,2)
        X=X+3*Y
        exp_sum=X.sum(axis=1,keepdims=True)
        X=X/exp_sum
        X=torch.from_numpy(X.astype(np.float32)).clone().requires_grad_()
        Y=torch.from_numpy(Y.astype(np.float32)).clone()
        loss=loss_func(X,Y)
        loss.backward()
        print(f"==================  epoch {e}  ===================")
        print(f"X = \n{X}")
        print(f"Y = \n{Y}")
        print(f"-----  損失 -----")
        for loss_name,value in loss_func.temporary_log.items():
            print(f"{loss_name} {value.item()}")
        print(f"-----  勾配  -----")
        print(f"X の勾配")
        print(X.grad)
        print(f"Y の勾配")
        print(Y.grad)