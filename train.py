import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from src.CNN import simplecnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transformer = transforms.Compose([
    transforms.ToTensor(),#图片转换为tensor张量
    transforms.Normalize((0.1307,), (0.3081,))

])

test_transformer = transforms.Compose([
    transforms.ToTensor(),  # 图片转换为tensor张量
    transforms.Normalize((0.1307,), (0.3081,))

])

#加载训练集和数据集
# 加载MNIST数据集
trainset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=train_transformer
)

testset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=test_transformer
)

#定义训练集加载器
train_loader = DataLoader(trainset, batch_size=32, num_workers=0, shuffle=True)#trainset传入的训练集， batch为批次训练的图像数量

test_loader = DataLoader(testset, batch_size=32, num_workers=0, shuffle=False)

def train(model, train_loader, criterion, optimizer, num_epoch):
    best_acc = 0.0
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for input,labels in tqdm(train_loader, desc=f"epoch {epoch + 1}/{num_epoch}",unit="batch"):#训练的时候可以看到epoch和batch
            inputs,labels = input.to(device),labels.to(device)#将数据传到设备上面
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0) #loss乘上批次大小，得到批次的loss
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"epoch {epoch + 1}/{num_epoch}, Train_loss: {epoch_loss:.4f}")

        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            os.makedirs("model_pth", exist_ok=True)
            save_model(model, save_path="model_pth/best.pth")
            print("model saved with best acc", best_acc)


def evaluate(model, test_loader, criterion):
    model.eval() #定义为验证模式
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  #评估模式下面不需要梯度计算
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) #将数据送到设杯里面
            outputs = model(inputs)
            loss = criterion(outputs, labels) #计算损失
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1) #获取模型预测的最大值
            total += labels.size(0) #计算总样本的数量
            correct += (predicted == labels).sum().item()#正确样本数累加

    avg_loss = test_loss / len(test_loader.dataset) #平均loss
    accuracy = 100.0* correct / total #计算准确率
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return accuracy

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

if __name__=='__main__':
    num_epochs = 10
    learning_rate = 0.001
    num_class =10
    save_path = r"model_pth/best.pth"
    model = simplecnn(num_class).to(device)# 实例化
    criterion = nn.CrossEntropyLoss() #指定损失函数为交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, criterion, optimizer, num_epochs)
    evaluate(model, test_loader, criterion)