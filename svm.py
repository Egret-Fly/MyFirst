from svmutil import *

y,x =svm_read_problem('F:\新建文件夹\dataset/train.txt')
yt,xt=svm_read_problem('F:\新建文件夹\dataset/test.txt')
model = svm_train(y,x)

p_label, p_acc, p_val = svm_predict(yt[0:117], xt[0:117], model)
print(p_label, p_acc, p_val)


devdevdev