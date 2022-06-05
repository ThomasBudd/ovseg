import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
plt.close()

# %%
mu0, mu1 = -1, 1
sigma2 = 1
p = lambda x: 1/np.sqrt(2*np.pi*sigma2) * np.exp(-1*x**2/2/sigma2)
p0, p1 = lambda x: p(x-mu0), lambda x:p(x-mu1)

x = np.linspace(-3,3,100)

plt.subplot(2,1,1)
plt.plot(x,p0(x), 'r', x,p1(x), 'b')
plt.legend(['p0', 'p1'])
plt.title('Densities')
plt.subplot(2,1,2)
Y_prob = p1(x)/(p0(x)+p1(x))
plt.plot(x, Y_prob, 'g')
plt.title('Classification probability')

# %%
n = 1000
n_iter = 1000
bs = 1000
n_ens = 7
n_warm = 50
w_list = [2]

X0 = np.sqrt(sigma2) * np.random.randn(n) + mu0
X1 = np.sqrt(sigma2) * np.random.randn(n) + mu1

X = np.concatenate([X0, X1]).reshape((-1,1))
Y = np.concatenate([np.zeros_like(X0), np.ones_like(X1)])

# data
X_gpu, Y_gpu = torch.from_numpy(X).cuda().type(torch.float), torch.from_numpy(Y).cuda().type(torch.long)

# loss
# loss_fctn = torch.nn.CrossEntropyLoss()

class Loss(torch.nn.Module):
    
    def __init__(self, w_list):
        super().__init__()
        weight = torch.tensor([1] + [np.exp(-1*w) for w in w_list]).type(torch.float).cuda()
        
        self.ce = torch.nn.CrossEntropyLoss(weight)
    
        self.weight_dice = 1 / (1 + np.exp(-1*w_list[0]))
    
    def forward(self, logs, target):
        
        pred = torch.nn.functional.softmax(logs, 1)[:, 1]
        dice = (target*pred + 1e-5) / (self.weight_dice * target + (1-self.weight_dice) * pred + 1e-5)
        return 1 - 1 * dice.mean() + self.ce(logs, target)


W = np.linspace(-3, 3, n_ens)

Y_prd = np.zeros((len(x), len(W)))

all_inds = list(range(2000))

for j, w in tqdm(enumerate(W)):
    w_list = [w]
    loss_fctn = Loss(w_list)
    
    # model for logits
    model = torch.nn.Linear(1,2).cuda()
    torch.nn.init.zeros_(model.bias)
    torch.nn.init.zeros_(model.weight)
        
    # optimizer
    lr_max = 1e-2
    lr_min = 1e-6
    opt = torch.optim.SGD(model.parameters(), lr=lr_min)
    
    model.train()
    for i in range(n_warm):
        ind = np.random.choice(all_inds, bs, replace=False)
        loss = loss_fctn(model(X_gpu[ind]), Y_gpu[ind])
        loss.backward()
        opt.step()
        lr = (lr_max - lr_min) * i/n_warm 
        opt.param_groups[0]['lr'] = lr
    for i in range(n_iter):
        
        ind = np.random.choice(all_inds, bs, replace=False)
        loss = loss_fctn(model(X_gpu[ind]), Y_gpu[ind])
        loss.backward()
        opt.step()
        lr = lr_max * np.cos(np.pi/2*(i/n_iter))
        opt.param_groups[0]['lr'] = lr
        # if i % 10 == 0:
            # print(f'{i}: {loss.detach().cpu().numpy():.4e} {model.weight[0][0].detach().cpu().numpy():.4e}')
    
    
    print(f'weight: {model.weight[0][0].detach().cpu().numpy():.4e}')
    print(f'bias: {model.bias[0].detach().cpu().numpy():.4e}')
    X_tst = torch.from_numpy(x).cuda().unsqueeze(1).type(torch.float)
    # Y_prd = torch.nn.functional.softmax(model(X_tst), 1).detach().cpu().numpy()[:, 1]
    Y_prd[:,j] = torch.nn.functional.softmax(model(X_tst), 1).detach().cpu().numpy()[:,1]
    
    plt.subplot(2, 1, 2)
    # plt.plot(x, Y_prd, 'b-')
    # compute accuracy


plt.subplot(2, 1, 2)
plt.plot(x, np.mean(Y_prd, 1), 'm')

Y_prob_hat = np.mean(Y_prd, 1)
R2 = 1 - np.sum((Y_prob_hat - Y_prob)**2) / np.sum((Y_prob - np.mean(Y_prob))**2)
print(f'R2: {100*R2:.3f}')

# %% Again with w=0

W = np.zeros(n_ens)

Y_prd = np.zeros((len(x), len(W)))

for j, w in tqdm(enumerate(W)):
    w_list = [w]
    loss_fctn = Loss(w_list)
    
    # model for logits
    model = torch.nn.Linear(1,2).cuda()
    torch.nn.init.zeros_(model.bias)
    torch.nn.init.zeros_(model.weight)
        
    # optimizer
    lr_max = 1e-2
    lr_min = 1e-6
    opt = torch.optim.SGD(model.parameters(), lr=lr_min)
    
    model.train()
    for i in range(n_warm):
        ind = np.random.choice(all_inds, bs, replace=False)
        loss = loss_fctn(model(X_gpu[ind]), Y_gpu[ind])
        loss.backward()
        opt.step()
        lr = (lr_max - lr_min) * i/n_warm 
        opt.param_groups[0]['lr'] = lr
    for i in range(n_iter):
        
        ind = np.random.choice(all_inds, bs, replace=False)
        loss = loss_fctn(model(X_gpu[ind]), Y_gpu[ind])
        loss.backward()
        opt.step()
        lr = lr_max * np.cos(np.pi/2*(i/n_iter))
        opt.param_groups[0]['lr'] = lr
        # if i % 10 == 0:
            # print(f'{i}: {loss.detach().cpu().numpy():.4e} {model.weight[0][0].detach().cpu().numpy():.4e}')
    
    
    print(f'weight: {model.weight[0][0].detach().cpu().numpy():.4e}')
    print(f'bias: {model.bias[0].detach().cpu().numpy():.4e}')
    X_tst = torch.from_numpy(x).cuda().unsqueeze(1).type(torch.float)
    # Y_prd = torch.nn.functional.softmax(model(X_tst), 1).detach().cpu().numpy()[:, 1]
    Y_prd[:,j] = torch.nn.functional.softmax(model(X_tst), 1).detach().cpu().numpy()[:,1]
    
    plt.subplot(2, 1, 2)
    # plt.plot(x, Y_prd, 'b-')
    # compute accuracy


plt.subplot(2, 1, 2)
plt.plot(x, np.mean(Y_prd , 1), 'k')

Y_prob_hat = np.mean(Y_prd, 1)
R2 = 1 - np.sum((Y_prob_hat - Y_prob)**2) / np.sum((Y_prob - np.mean(Y_prob))**2)
print(f'R2: {100*R2:.3f}')

plt.legend(['Analytic', 'Calibrated ensemble', 'Uncalibrated ensemble'])
# %% Again with variable bias

# class Model(torch.nn.Module):
    
#     def __init__(self, n_hid=10):
#         super().__init__()

#         self.x_lin = torch.nn.Linear(1,2, bias=False)

#         # self.w_bias = torch.nn.Sequential(*[torch.nn.Linear(1, n_hid),
#         #                                     torch.nn.ReLU(),
#         #                                     torch.nn.Linear(n_hid, 1)])
        
#         self.w_bias = torch.nn.Linear(1,1)
#     def forward(self, x, w):
#         return self.x_lin(x) + self.w_bias(w)

# class Loss(torch.nn.Module):
    
#     def forward(self, logs, target, weights):
        
#         weight_dice = 1 / (1 + np.exp(-1*weights)).item()
#         pred = torch.nn.functional.softmax(logs, 1)[:, 1]
#         dice = (target*pred + 1e-5) / (weight_dice * target + (1-weight_dice) * pred + 1e-5)
#         dice_loss = 1 - 1*dice.mean()
        
#         # weights_ce = torch.stack([torch.ones_like(weights), torch.exp(-1*weights)], 1)
#         # ce_loss = 0
#         # for b in range(logs.shape[0]):
#         #     ce_loss += torch.nn.functional.cross_entropy(logs[b:b+1],
#         #                                                  target[b:b+1],
#         #                                                  weight=weights_ce[b])
#         # ce_loss /= logs.shape[0]
#         weights_ce = torch.tensor([1, np.exp(-weights)]).cuda().type(torch.float)
#         ce_loss = torch.nn.functional.cross_entropy(logs,
#                                                     target,
#                                                     weight=weights_ce)
        
#         return dice_loss + ce_loss

# model = Model().cuda()
# loss_fctn = Loss()

# lr_max = 1e-2
# lr_min = 1e-6
# n_iter = 1000
# n_warm = 50
# opt = torch.optim.SGD(model.parameters(), lr=lr_min)

# model.train()

# print('Warmup')
# for i in range(n_warm):
    
#     weights = np.random.rand()*6-3
#     weights_gpu = torch.tensor([[weights]]).cuda()
#     loss = loss_fctn(model(X_gpu, weights_gpu), Y_gpu, weights)
#     loss.backward()
#     opt.step()
#     lr = (lr_max - lr_min) * i/n_warm 
#     opt.param_groups[0]['lr'] = lr
    
# print('Training')  
# for i in range(n_iter):
    
#     weights = np.random.rand()*6-3
#     weights_gpu = torch.tensor([[weights]]).cuda()
#     loss = loss_fctn(model(X_gpu, weights_gpu), Y_gpu, weights)
#     loss.backward()
#     opt.step()
#     lr = lr_max * np.cos(np.pi/2*(i/n_iter))
#     opt.param_groups[0]['lr'] = lr
#     if i % 50 == 0:
#         print(f'{i}: {loss.detach().cpu().numpy():.4e}')
# # %%
# plt.plot(x, Y_prob, 'g')
# plt.title('Classification probability')
# Y_hat = torch.nn.functional.softmax(model(X_tst, torch.tensor([[2.0]]).cuda()), 1).detach().cpu().numpy()[:,1]
# plt.plot(x, Y_hat, 'm')
# Y_hat = torch.nn.functional.softmax(model(X_tst, torch.tensor([[-2.0]]).cuda()), 1).detach().cpu().numpy()[:,1]
# plt.plot(x, Y_hat, 'm')


# W = torch.linspace(-3, 3, 100).unsqueeze(1).cuda()
# biases = model.w_bias(W)
