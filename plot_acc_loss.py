# File: plot_acc_loss.py
import matplotlib.pyplot as plt
import json
import os

def plot_training_curves(result_dir):
    history_path = os.path.join(result_dir, 'training_history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(result_dir, 'training_curves.png'))
    plt.close()