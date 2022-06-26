import argparse
import random
from torch.utils.data import DataLoader
from torch import autograd
from datasets import *
from models import *
from draw import *


def gradient_penalty(D, xr, xf, batchsz, args):

    # [b, 1]
    t = torch.rand(batchsz, 1).to(args.device)
    # [b, 1] => [b, 10]
    t = t.expand_as(xr)
    # interpolation
    mid = t * xr + (1 - t) * xf
    # set it requires gradient
    mid.requires_grad_()

    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True,
                          only_inputs=True)[0]

    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

    return gp


def main(args):

    global loss_D, loss_G
    device = args.device

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build dataset and dataloader
    print("\nProcessing " + args.dataset + " dataset...")
    if args.dataset == "points":
        dataset = Points()
        train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
    print("Data processing finished!")

    # build model
    print("\nBuilding model " + args.model + "...")
    if args.model in ["GAN", "WGAN", "WGAN-GP"]:
        G = Generator()
        D = Discriminator()
    else:
        raise ValueError(f"dataset {args.model} not supported")
    G.to(device)
    D.to(device)
    print("Model building finished!")

    # set up optimizers
    if args.optimizer == 'adam':
        optim_G = torch.optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
        optim_D = torch.optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))
    elif args.optimizer == 'sgd':
        optim_G = torch.optim.Adam(G.parameters(), lr=3e-4)
        optim_D = torch.optim.Adam(D.parameters(), lr=3e-4)
    elif args.optimizer == 'rmsprop':
        optim_G = torch.optim.RMSprop(G.parameters(), lr=1e-4)
        optim_D = torch.optim.RMSprop(D.parameters(), lr=1e-4)
    else:
        raise ValueError(f"dataset {args.optimizer} not supported")

    # start train and validation
    print("\nStart training...")
    all_loss = []
    for epoch in range(1, args.epochs + 1):
        # train loss
        for data in train_loader:
            # 1. optimize Discriminator
            # 1.1 train real data
            xr = data.to(device)
            batchsz = xr.shape[0]
            predr = D(xr)
            # 1.2 train fake data
            z = torch.randn(batchsz, 10).to(device)
            xf = G(z).detach()
            predf = D(xf)

            # 1.3 loss and update Discriminator
            loss_D = - (torch.log(predr) + torch.log(1. - predf)).mean()

            if args.model == 'WGAN':
                for p in D.parameters():
                    # print(p.data)
                    p.data.clamp_(-args.CLAMP, args.CLAMP)

            if args.model == 'WGAN-GP':
                loss_D += 0.2 * gradient_penalty(D, xr, xf.detach(), batchsz, args)

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # 2. optimize Generator
            z = torch.randn(args.batch_size, 10).to(device)
            xf = G(z)
            predf = D(xf)
            loss_G = torch.log(1. - predf).mean()
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        if epoch % 5 == 0:
            print('[epoch %d/%d] Discriminator loss: %.3f, Generator loss: %.3f'
                  % (epoch, args.epochs, loss_D.item(), loss_G.item()))
            all_loss.append([loss_D.item(), loss_G.item()])
        if epoch % 30 == 0 and args.draw:
            input = torch.randn(1000, 10).to(device)
            output = G(input)
            output = output.to('cpu').detach()
            xy = np.array(output)
            draw_scatter(D, xy, epoch, args.model)

    # draw the loss
    all_loss = np.array(all_loss)
    x = np.arange(len(all_loss))
    y1 = all_loss[:, 0]
    y2 = all_loss[:, 1]
    fig = plt.figure(2, figsize=(16, 16), dpi=150)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(x, y1, 'r', label='loss_D')
    ax2.plot(x, y2, 'g', label='loss_G')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.savefig(args.output_path + args.model + "/loss.jpg")

    # save the model
    state = {"model_D": D.state_dict(), "model_G": G.state_dict()}
    torch.save(state, args.output_path + 'models/' + args.model + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("lab4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="GAN")  # GAN WGAN WGAN-GP
    parser.add_argument("--epochs", default=1000)
    parser.add_argument("--batch-size", default=2000)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--dataset", default="points")  # points
    parser.add_argument("--output-path", default="./result/")
    parser.add_argument("--hidden-size", default=128)
    parser.add_argument("--input-size", default=128)
    parser.add_argument("--CLAMP", default=0.1)
    parser.add_argument("--optimizer", default="rmsprop")  # adam sgd rmsprop
    parser.add_argument("--draw", default=False, help="draw the loss and process")

    args = parser.parse_args()
    print(args)

    main(args)
