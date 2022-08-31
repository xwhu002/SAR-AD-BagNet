from __future__ import print_function
import os
import argparse
import save_1
from torchvision import datasets, transforms
import BagNet18
from models.wideresnet import *
from models.resnet import *
from trades import trades_loss
import time
parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=30, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--mean',type=float,default=0.184,help='mean of dataset')
parser.add_argument('--std',type=float,default=0.119,help='standard deviation of dataset')
global args
args = parser.parse_args()
def train(train_iter, test_iter, net, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)

    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for i,(X, y) in enumerate(train_iter):
            X=X.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            l = trades_loss(model=net,
                               x_natural=X,
                               y=y,
                               optimizer=optimizer,
                               step_size=args.step_size,
                               epsilon=args.epsilon,
                               perturb_steps=args.num_steps,
                               beta=args.beta,
                               distance='l_inf')

            l.backward()
            optimizer.step()
            y_hat = net(X)
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch {}, loss {:.4f}, train acc {:.4f}, test acc{:.4f}, time {:.2f} sec'
              .format(epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        save_1.save_model_w_condition(model=net,
                                    model_dir='D:/pythondata/TRADES-master/saved_models/',
                                    model_name=str(epoch+1) + 'bagnet_TR_8_2', accu=test_acc,
                                    target_accu=0.98)
        #save.save_model_w_condition(model=net, model_dir='D:/pythondata/TRADES-master/saved_models/',
                                    #model_name=str(epoch) + 'bagnet_TR_8_2', accu=test_acc,epoch=epoch)
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):

                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n
# settings
if __name__ == '__main__':
    mean = args.mean
    std = args.std
    img_size=100
    train_dir = '.../train'
    test_dir = '.../val'
    train_batch_size = args.batch_size
    test_batch_size = args.batch_size
    # initialize the VGG model for RGB images with 3 channels
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=0, pin_memory=False)
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_dir = '.../saved_models/'
    lr = 0.001
    num_epochs=args.epochs
    net = bagnet18_1.BagNet18(pretrained=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(train_loader,test_loader,net,optimizer,device,num_epochs)
    torch.save(obj=net, f=os.path.join(mode_dir, 'Bagnet_trades ' + '.pth'))