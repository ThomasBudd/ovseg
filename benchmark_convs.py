from time import perf_counter
import torch



def benchmark(net, xb):
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            
            for _ in range(100):
                net(xb)
            
            st = perf_counter()
            
            for _ in range(1000):
                net(xb)
            
            et = perf_counter()
            
            print('{:.4e}s elapsed'.format(et-st))

N = 256
F = 32

for i in range(5):
    n = 256 // 2 ** i
    f = F * 2 ** i
    
    xb = torch.zeros((1, f, n, n)).cuda()
    net = torch.nn.Conv2d(in_channels=f, out_channels=f, kernel_size=3, padding=1).cuda()
    
    print('n={}, f={}'.format(n, f))
    benchmark(net, xb)
