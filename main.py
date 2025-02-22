# File: main.py (增强版)
import argparse
import json
import os
from datetime import datetime
import torch
from torch.optim import AdamW
from model.DBDRSNet import DB_DRSN1D
from ulti.DataLoad import get_loaders
from ulti.acc import calculate_metrics
from ulti.init_seed import init_seed

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化种子
        init_seed(args.seed)
        
        # 创建结果目录
        self.result_dir = os.path.join('result', args.dataset, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 数据加载
        self.train_loader, self.val_loader = get_loaders(
            os.path.join('dataset', args.dataset), 
            batch_size=args.batch_size
        )
        
        # 模型初始化
        self.model = DB_DRSN1D(num_classes=len(self.train_loader.dataset.classes)).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 训练记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss/len(self.train_loader), 100*correct/total
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss/len(self.val_loader), 100*correct/total
    
    def run(self):
        best_acc = 0.0
        for epoch in range(1, self.args.epochs+1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.result_dir, 'best_model.pth'))
            
            # 打印进度
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            # 定期保存
            if epoch % self.args.save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(self.result_dir, f'epoch_{epoch}.pth'))
        
        # 保存训练记录
        with open(os.path.join(self.result_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f)
        
        # 绘制曲线
        plot_training_curves(self.result_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DB-DRSN Training')
    parser.add_argument('--dataset', type=str, default='CWRU', choices=['CWRU', 'HUST', 'PN'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_interval', type=int, default=10)
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.run()