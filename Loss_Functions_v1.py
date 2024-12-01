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
    def __init__(self,loss_text):
        dammy=loss_text
        dammy=re.sub("([\(\)])"," \\1 ",dammy)
        dammy=re.sub("([\+\-\*\/]+)"," \\1 ",dammy)
        self.token_list=[i for i in re.split(" +",dammy) if i!=""]
        self.temporary_log={"All":None}#損失関数ごとに平均をとるためのもの
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
        new_token_list.extend(stack[::-1])
        self.token_list=new_token_list
    def token_check(self,token):
        #演算子,括弧,数値,関数に分類する
        if token in Loss_Functions.operator_list:
            return "operator"
        try:
            float(token)
        except ValueError:
            if token.isalnum():
                return "loss"
        else:
            return "numelic"
        
        return "Error"
    #損失計算
    def __call__(self,X=None,y=None):
        #temporary_logの初期化
        self.temporary_log={loss_name:[] for loss_name in self.temporary_log}

        workspace=[]
        for token in self.token_list:
            token_check=self.token_check(token)
            if token_check=="numelic":
                workspace.append(float(token))
            elif token_check=="loss":
                #tokenで指定して辞書から損失関数をもらう。ないときはL2ロスをもらう
                loss_part=Loss_dict.get(token,nn.MSELoss())
                loss_value=loss_part(X,y)
                workspace.append(loss_value)
                self.temporary_log[token]=loss_value
            elif token_check=="operator":
                operator=Loss_Functions.operators[token]
                #演算子が必要とする個数の値をpopする
                input=[workspace.pop() for i in range(operator.need_term)]#このままでは順番が逆になっている
                workspace.append(operator(input[::-1]))
        loss=workspace[0]#tensor
        self.temporary_log["All"]=loss
        #data形式は{"loss_name1":loss_value1,"loss_name2":loss_value2,...}
        return loss

if __name__=="__main__":
    text="2*(L1+L2)"
    x=torch.randn(5,5,requires_grad=True)
    y=torch.randn(5,5)
    Loss=Loss_Functions(text)
    print(f"損失関数\n{text}")
    print(f"逆ポーランド\n{Loss.token_list}")
    loss=Loss(x,y)
    print(f"損失値:{loss}")
    loss.backward()
    print("-----勾配-----")
    print(f"{x.grad} ")
    print(f"{y.grad} ")

    
    