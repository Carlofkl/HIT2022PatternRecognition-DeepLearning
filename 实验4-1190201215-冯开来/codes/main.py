import argparse
import random
import time
import os
from sklearn.metrics import classification_report
from datasets import *
from models import *
from engine import *


shopping_names = ['书籍', '平板', '手机', '水果', '洗发水', '热水器', '蒙牛', '衣服', '计算机', '酒店']  # 全部类别


def main(args):
    # device
    # global test_loader
    device = args.device

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build dataset and dataloader
    print("\nProcessing " + args.dataset + " dataset...")
    if args.dataset == "shopping":
        train_loader, val_loader, test_loader = build_shopping(args)
        output_size = 10
    elif args.dataset == "climate":
        train_loader, test_loader = build_climate(args)
        val_loader = None
        output_size = 1
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
    print("Data processing finished!")

    # build model
    print("\nBuilding model " + args.model + "...")
    if args.model == "RNN":
        my_model = RNN(args, output_size)
    elif args.model == "GRU":
        my_model = GRU(args, output_size)
    elif args.model == "LSTM":
        my_model = LSTM(args, output_size)
    elif args.model == "Bi-LSTM":
        my_model = LSTM(args, output_size,  bidirectional=True)
    else:
        raise ValueError(f"model {args.model} not supported")
    my_model.to(device)
    print("Model building finished!")

    # set up optimizers
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=my_model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(params=my_model.parameters(), lr=0.0001)
    else:
        raise ValueError(f"optimizer {args.optimizer} not supported")

    # run model only on test
    if args.test and args.dataset == 'climate':
        print("Testing temperature by " + args.model + "...")
        filename = "best_" + args.model + "_climate.pth"
        load_path = os.path.join(args.output_path, "models/", filename)
        checkpoint = torch.load(load_path)
        my_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        my_model.eval()
        test_climate(my_model, test_loader, args)
        return
    if args.test:
        print("Testing accuracy, recall, F1 by " + args.model + "...")
        filename = "best_" + args.model + ".pth"
        load_path = os.path.join(args.output_path, "models/", filename)
        checkpoint = torch.load(load_path)
        my_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        my_model.eval()

        preds = []
        true = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                label, sentence = data['cat'].to(args.device), data['review'].to(args.device)
                output = my_model(sentence)
                pred = torch.argmax(output, dim=1)
                pred, label = pred.to('cpu').item(), label.to('cpu').item()
                preds.append(pred)
                true.append(label)
        result = classification_report(true, preds, target_names=shopping_names)
        print(result)
        return

    # start train and validation
    all_loss = []
    best_accuracy = 0.0
    accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        print("Epoch [{}/{}]".format(epoch, args.epochs))
        epoch_loss = train(my_model, train_loader, optimizer, epoch, args)
        if val_loader:
            accuracy = validation(my_model, val_loader, args)

        all_loss.extend(epoch_loss)
        end = time.time()
        print('Epoch %d finished, took %.2fs' % (epoch, end - start))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            filename = "best_" + args.model + ".pth"
            checkpoint_path = os.path.join(args.output_path, "models/", filename)
            torch.save(
                {
                    "model": my_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args
                },
                checkpoint_path
            )
        if not val_loader:
            filename = "best_" + args.model + "_climate.pth"
            checkpoint_path = os.path.join(args.output_path, "models/", filename)
            torch.save(
                {
                    "model": my_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args
                },
                checkpoint_path
            )

    # draw the loss
    x = np.arange(len(all_loss))
    plt.plot(x, all_loss, 'r')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser("lab4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="GRU")  # RNN GRU LSTM Bi-LSTM
    parser.add_argument("--epochs", default=2)
    parser.add_argument("--batch-size", default=30)  # 本次实验没有用到过
    parser.add_argument("--seed", default=42)
    parser.add_argument("--dataset", default="shopping")  # shopping or climate
    parser.add_argument("--output-path", default="./result/")
    parser.add_argument("--hidden-size", default=128)
    parser.add_argument("--input-size", default=128, type=int)  # if climate, only be 5
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--test", action="store_true", help="Only run test")

    args = parser.parse_args()
    print(args)

    main(args)

    # run climate
    # python main.py --input-size 5