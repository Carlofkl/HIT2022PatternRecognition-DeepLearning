import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()


def train(model, data_loader, optimizer, epoch, args):
    model.train()
    train_loss = 0.0
    all_loss = []
    for idx, data in enumerate(data_loader):
        label, sentence = data['cat'].to(args.device), data['review'].to(args.device)
        print(sentence.shape)
        if len(sentence.shape) < 3:
            sentence = sentence[None]  # expand for batchsz
        # print(sentence.shape)  # 1, words, 128
        output = model(sentence)

        optimizer.zero_grad()
        if args.dataset == 'climate':
            loss = abs(output - label)
        else:
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if idx % 376 == 0 and idx > 0:
            all_loss.append(train_loss / 376)
            print('[epoch %d, batch %d] loss: %.3f' % (epoch, idx, train_loss / 376))
            train_loss = 0.0
    return all_loss


def validation(model, data_loader, args):
    model.eval()
    model.zero_grad()
    correct_num = 0
    val_num = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            label, sentence = data['cat'].to(args.device), data['review'].to(args.device)
            output = model(sentence)
            pred = torch.argmax(output, dim=1)
            correct_num += (pred == label).sum().item()
            val_num += 1
    print('Accuracy on validation set: %.2f%%\n' % (100.0 * correct_num / val_num))
    return correct_num / val_num


def test_climate(model, data_loader, args):
    trues = []
    preds = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            label, sentence = data['cat'].to(args.device), data['review'].to(args.device)
            if len(sentence.shape) < 3:
                sentence = sentence[None]
            pred = model(sentence)
            pred, label = pred.to('cpu').item(), label.to('cpu').item()
            trues.append(float(label))
            preds.append(float(pred))

    for i in range(10):
        start = 144 * i
        end = 144 * i + 288
        loss = []
        for j in range(start, end):
            loss.append(abs(trues[j] - preds[j]))
        mean_loss = np.mean(loss)
        median_loss = np.median(loss)
        print('[Week %d] Mean-loss: %.3f, Median-loss: %.3f'
                  % (i, mean_loss, median_loss))

        x = np.arange(288)
        plt.figure(1, (16, 8), 100)
        plt.cla()
        plt.plot(x, trues[start: end], 'r')
        plt.plot(x, preds[start: end], 'g--')
        plt.legend(["true", "predict"], loc='upper left')
        plt.savefig('./result/pred-true-' + str(i) + '.jpg')



