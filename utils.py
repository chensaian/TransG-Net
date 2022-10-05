import sys
import numpy as np
import torch
from tqdm import tqdm

def train_one_epoch_ms(model, optimizer, data_loader, device, epoch):
    model.train()

    loss_function = torch.nn.MSELoss()

    accu_loss = torch.zeros(1) # 累计损失
    accu_loss = accu_loss.to(device)

    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)

    for data in data_loader:
        ms_data, labels = data.ms_spec, data.y
        sample_num += ms_data.shape[0]
        pred = model(data.to(device))

        labels = torch.tensor(labels)

        pred=pred.view(-1)
        
        loss = loss_function(pred.float(), labels.float().to(device))
        loss.backward()
        accu_loss += loss.clone().detach()
        
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (sample_num))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / sample_num


@torch.no_grad()
def evaluate_ms(model, data_loader, device, epoch):
    model.eval()
    
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for data in data_loader:
        ms_data, labels = data.ms_spec, data.y
        sample_num += ms_data.shape[0]

        pred = model(data.to(device))
        labels = [np.argmax(item) for item in labels]
        labels = torch.tensor(labels)


        pred=pred.view(-1)

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (sample_num))
                                                                               

    return accu_loss.item() / sample_num
