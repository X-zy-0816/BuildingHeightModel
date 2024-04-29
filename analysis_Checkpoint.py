import torch
import matplotlib.pyplot as plt

# 加载模型
model_path = 'runs/tlcnetu_zy3bh/finetune.tar'
model = torch.load(model_path)

print(model['epoch'])





#
# # 提取训练过程中的指标
# train_loss = model['train_loss']
# test_loss = model['test_loss']
# accuracy = model['accuracy']
# rmse = model['rmse']
#
# # 可视化训练过程中的损失
# plt.figure()
# plt.plot(train_loss, label='Train Loss')
# plt.plot(test_loss, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 可视化准确率和 RMSE
# plt.figure()
# plt.plot(accuracy, label='Accuracy')
# plt.plot(rmse, label='RMSE')
# plt.xlabel('Epoch')
# plt.ylabel('Metric')
# plt.legend()
# plt.show()
