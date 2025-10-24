import sys
import os
sys.path.append(os.path.abspath('.'))  # 保证能找到 data_provider 目录

from data_provider.data_factory import data_provider

class Args:
    root_path = './dataset/ETT-small/'
    data_path = 'ETTh1.csv'
    data = 'ETTh1'
    features = 'M'
    target = 'OT'
    embed = 'timeF'
    percent = 100
    seasonal_patterns = None
    freq = 'h'
    batch_size = 4
    num_workers = 0
    seq_len = 24
    label_len = 12
    pred_len = 12

if __name__ == "__main__":
    args = Args()
    flag = 'train'  # 可选 'train', 'val', 'test'
    dataset, dataloader = data_provider(args, flag)

    print(f"数据集长度: {len(dataset)}")
    print("第一个样本：")
    sample = dataset[0]
    print("seq_x shape:", sample[0].shape)
    print("seq_y shape:", sample[1].shape)
    print("seq_x_mark shape:", sample[2].shape)
    print("seq_y_mark shape:", sample[3].shape)

    print("\nDataLoader 批次：")
    for i, batch in enumerate(dataloader):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        print(f"Batch {i}: batch_x.shape={batch_x.shape}, batch_y.shape={batch_y.shape}")
        if i >= 2:
            break   # 只看前3个 batch