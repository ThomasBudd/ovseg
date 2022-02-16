import torch
from time import perf_counter

class SeperateConv(torch.nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.conv1 = torch.nn.Conv3d(self.in_ch, self.out_ch,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1),
                                     bias=False)
        
        self.conv2 = torch.nn.Conv3d(self.out_ch, self.out_ch,
                                     kernel_size=(3, 1, 1),
                                     padding=(1,0,0),
                                     bias=False)
    
    def forward(self, xb):
        return self.conv2(self.conv1(xb))


convs = [torch.nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False).cuda(),
         SeperateConv(64, 64).cuda()]

xb = torch.zeros((1, 64, 32, 64, 64)).cuda()


def benchmark(conv, xb):
    with torch.no_grad():
        
        for _ in range(20):
            out = conv(xb)
        
        t_start = perf_counter()
        
        for _ in range(100):
            out = conv(xb)
        
        t_end = perf_counter()
        
        print('{:.6f}s ellapsed for 100 runs'.format(t_end-t_start))

for conv in convs:
    benchmark(conv, xb)
