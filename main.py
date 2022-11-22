import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import time

from ResNet import *
import hiddenlayer as hl
from tqdm import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
MODELS = ['resnet_O', 'resnet_N', 'resnet_B', 'resnet_C', 'resnet_P']


# (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
# (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

# # cifar10
# train_set = datasets.CIFAR10(root='../data/', train=True, download=True, transform=train_transform)
# print(train_set.train_data.shape)
# print(train_set.train_data.mean(axis=(0,1,2))/255)
# print(train_set.train_data.std(axis=(0,1,2))/255)
# # (50000, 32, 32, 3)
# # [0.49139968  0.48215841  0.44653091]
# # [0.24703223  0.24348513  0.26158784]
# CIFAR-10:  
# (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
# Data

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])


# def model_modifiy(model_name):
#     print('Building model..')
#     if model_name == 'vgg_16':
#         model = models.vgg16(pretrained=False)
#         model.classifier=nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),nn.ReLU(inplace=True),nn.Dropout(p=0.5, inplace=False),nn.Linear(in_features=4096,out_features=10,bias=True))
#         return model
#     if model_name == 'vgg_19':
#         model = models.vgg19(pretrained=False)
#         model.classifier=nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),nn.ReLU(inplace=True),nn.Dropout(p=0.5, inplace=False),nn.Linear(in_features=4096,out_features=10,bias=True))
#         return model
#     if model_name == 'resnet_18':
#         model = models.resnet18(pretrained=False)
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, 10)
#         return model
#     if model_name == 'resnet_50':
#         model = models.resnet50(pretrained=False)
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, 10)
#         return model
#     if model_name == 'resnet_101':
#         model = models.resnet101(pretrained=False)
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, 10)
#         return model
#     if model_name == 'mobilenet_v2':
#         model = models.mobilenet_v2(pretrained=False)
#         # num_ftrs = model.classifier.in_features
#         model.classifier=nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features=1280,out_features=10,bias=True))
#         return model
#     if model_name == 'densenet121':
#         model = models.densenet121(pretrained=False)
#         num_ftrs = model.classifier.in_features
#         model.classifier = nn.Linear(num_ftrs, 10, bias=True)
#         return model

def model_fit(model_name):
    if model_name == 'resnet_O':
        return resnet_O()
    elif model_name == 'resnet_F':
        return resnet_F()
    elif model_name == 'resnet_N':
        return resnet_N()
    elif model_name == 'resnet_B':
        return resnet_B()
    elif model_name == 'resnet_C':
        return resnet_C()
    elif model_name == 'resnet_P':
        return resnet_P()
    elif model_name == 'resnet_K':
        return resnet_K()
    else:
        return resnet_O()
    


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run(model_name, l1_enable=False, Adam_enable=False, redu_enable=False, resume=False ,N_EPOCHS = 60, l1_lambda = 0.001, l2_alpha = 5e-4, learning_rate = 1e-2, BATCH_SIZE = 128):
    best_acc = 0
    start_epoch = 0

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    print('Batch_size: ' + str(BATCH_SIZE))
    print('Train_size:' + str(len(train_loader)))
    print('Test_size:' + str(len(test_loader)))


    model = model_fit(model_name)
    net = model.to(DEVICE)
    history=hl.History()
    canvas=hl.Canvas()

    if DEVICE == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if resume:
        print('Loading snapshot..')
        assert os.path.isdir('snapshot'), 'Error: no snapshot directory found!'
        snapshot = torch.load('./snapshot/' + model_name + '_state.pth')
        net.load_state_dict(snapshot['net'])
        best_acc = snapshot['acc']
        start_epoch = snapshot['epoch'] + 1
        print('Loading history..')
        assert os.path.isdir('history'), 'Error: no history directory found!'
        history.load('./history/'+model_name+"_history.pkl")
        history.summary()

    criterion = nn.CrossEntropyLoss()
    # CrossEntropyLoss() already has softmax, so there is no need to add softmax layer at the end of net.
    

    if Adam_enable:
        optimizer = optim.Adam(net.parameters(),lr=learning_rate,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
    else:
        if l1_enable:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_alpha)
        
    
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num)
    print('Trainable: ', trainable_num)

    if redu_enable:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    

    for epoch in range(start_epoch, start_epoch+N_EPOCHS):
        start_time = time.time()
        print('Training...')
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        print('Testing...')
        test_loss, test_acc = test(model, test_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss: {train_loss}')
        print(f'\t Train Diff: {train_acc}%')
        print(f'\t Test Loss: {test_loss}')
        print(f'\t Test Diff: {test_acc}%')
        print('\n')

        history.log(epoch,train_loss=train_loss,train_acc=train_acc,test_loss=test_loss,test_acc=test_acc)
        history.progress()
        with canvas:
            canvas.draw_plot([history['train_loss'],history['test_loss']])
            canvas.draw_plot([history['train_acc'],history['test_acc']])

        if test_acc > best_acc:
            print('Saving model...')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('./snapshot'):
                os.mkdir('./snapshot')
            if not os.path.isdir('./figure'):
                os.mkdir('./figure')
            if not os.path.isdir('./history'):
                os.mkdir('./history')
            torch.save(state, './snapshot/' + model_name + '_state.pth')
            canvas.save('./figure/'+model_name+"_progress.png")
            history.save('./history/'+model_name+"_history.pkl")
            best_acc = test_acc

        scheduler.step()
    print('End.')
        


def train(net, iterator, optimizer, criterion, l1_enable = False):
    correct = 0
    total = 0
    epoch_loss = 0
    epoch_acc = 0
    net.train()
    for [batch_inputs, batch_targets] in tqdm(iterator):
        optimizer.zero_grad()
        batch_outputs = net(batch_inputs.to(DEVICE))
        #   print(batch_inputs)
        #   print(batch_inputs.dtype)
        loss = criterion(batch_outputs,batch_targets.to(DEVICE))

        if l1_enable:
            l1_lambda = 0.001
            l1_norm = sum(torch.linalg.norm(p, 1) for p in net.parameters())
            loss = loss + l1_lambda * l1_norm
        #   print(batch_targets)
        #   print(batch_targets.dtype)
        #   print('')
        loss.backward()
        optimizer.step()

        _, batch_predictions = batch_outputs.max(1)
        total += batch_targets.size(0)
        correct += batch_predictions.cpu().eq(batch_targets.cpu()).sum().item()

        epoch_acc = 100.*correct/total
        epoch_loss = epoch_loss + loss.item()
  
    return epoch_loss / total, epoch_acc

def test(net, iterator, criterion):
    correct = 0
    total = 0
    epoch_loss = 0
    epoch_acc = 0
    net.eval()
    with torch.no_grad():
        for [batch_inputs, batch_targets] in tqdm(iterator):
            batch_outputs = net(batch_inputs.to(DEVICE))
            loss = criterion(batch_outputs,batch_targets.to(DEVICE))

            _, batch_predictions = batch_outputs.max(1)
            total += batch_targets.size(0)
            correct += batch_predictions.cpu().eq(batch_targets.cpu()).sum().item()

            epoch_acc = 100.*correct/total
            epoch_loss = epoch_loss + loss.item()
        
    return epoch_loss / total, epoch_acc






if __name__ == '__main__':
    # N_EPOCHS = 60
    # l1_lambda = 0.001
    # l2_alpha = 5e-4
    # learning_rate = 1e-2
    # BATCH_SIZE = 128

    # for m in MODELS:
    #     run(model_name=m,resume=True)
    # run(model_name='resnet_O')
    # run(model_name='train1', BATCH_SIZE=328)
    run(model_name='train2', Adam_enable=True)
    run(model_name='train3', l1_enable=True)
    run(model_name='train4', redu_enable=True)
    run(model_name='train5', learning_rate=1e-4)
    
