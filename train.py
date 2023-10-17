from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

from loss import yolo_loss
import torch.optim as optim
from model import Yolov1
import config
from dataset import load_data_to_train

DEVICE = "cuda" if torch.cuda.is_available else "cpu"


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(out_W=config.out_W, out_H=config.out_H, num_boxes=config.box_per_cell, num_classes=config.nclass).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=0
    )
    

    x_train, y_train = load_data_to_train()

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)

    # Tạo một TensorDataset từ x_train và y_train
    dataset = TensorDataset(x_train, y_train)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(config.epochs):
        train_fn(train_loader, model, optimizer, loss_fn=yolo_loss)

if __name__ == "__main__":
    main()





