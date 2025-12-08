# Taken from https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py
# Uncompiled variant of airbench94_compiled.py
# Optimized for H100 Speedrun Experiments

import os
import sys
import uuid
import random
import numpy as np
from math import ceil
import time

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# --- H100 SYSTEM OPTIMIZATIONS ---
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- EXPERIMENT CONFIGURATION (EDIT THIS PER RUN) ---
# Config A (Baseline):      BS=1024, WIDTH=1.0, GROUPS=False
# Config B (Batch Speed):   BS=4096, WIDTH=1.0, GROUPS=False
# Config C (Free Lunch):    BS=1024, WIDTH=2.0, GROUPS=False
# Config D (Architecture):  BS=1024, WIDTH=1.0, GROUPS=True

EXPERIMENT_NAME = "h100_4096_2.0"
BATCH_SIZE = 4096
WIDTH_MULTIPLIER = 1
USE_GROUPED_CONV = False
N_RUNS = 100
# ----------------------------------------------------

# --- FIX 1: Helper function for reproducibility ---
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

hyp = {
    'opt': {
        'train_epochs': 9.9,
        'batch_size': BATCH_SIZE,
        'lr': 46.0,
        'momentum': 0.85,
        'weight_decay': 0.0153,
        'bias_scaler': 64.0,
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,
    },
}

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    
    # H100 Optimization: Explicitly request channels_last
    images_out = torch.empty((len(images), 3, crop_size, crop_size), 
                           device=images.device, 
                           dtype=images.dtype,
                           memory_format=torch.channels_last)
                           
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), 
                               device=images.device, 
                               dtype=images.dtype,
                               memory_format=torch.channels_last)
                               
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, gpu=0):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location='cpu')
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        img_dtype = torch.float16 if device.type == 'cuda' else torch.float32

        # H100 Optimization: Ensure channels_last on load
        self.images = (self.images.to(img_dtype) / 255).permute(0, 3, 1, 2).to(
            memory_format=torch.channels_last
        )

        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
        self.normalize = T.Normalize(CIFAR_MEAN.to(device), CIFAR_STD.to(device))
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class Conv(nn.Conv2d):
    # Added groups parameter for architecture ablations
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias, groups=groups)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        # Stage 1: Standard Conv (Expand channels) - Always dense
        self.conv1 = Conv(channels_in,  channels_out, groups=1)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        
        # H100 Speedup: ReLU is faster than GELU
        self.activ = nn.ReLU(inplace=True)

        # Stage 2: Toggleable Grouped Conv
        # If enabled, uses groups=4 (ResNeXt-style efficiency)
        g = 4 if USE_GROUPED_CONV else 1
        
        self.conv2 = Conv(channels_out, channels_out, groups=g)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

def make_net():
    base_widths = hyp['net']['widths']
    # Apply Width Multiplier for Saturation Experiment
    widths = {
        k: max(8, int(v * WIDTH_MULTIPLIER))
        for k, v in base_widths.items()
    }
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(), # Keep first GELU for stability
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    
    if device.type == 'cuda':
        net = net.half()
    net = net.to(device)
    
    # H100 Optimization: Force channels_last
    net = net.to(memory_format=torch.channels_last)

    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}
    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)

logging_columns_list = ['run', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()
    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))
    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

def main(run):
    if isinstance(run, int):
        set_seed(run)

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale 
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'])
    
    if run == 'warmup':
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    
    total_train_steps = ceil(len(train_loader) * epochs)
    
    # --- FIX START: Order of Operations ---
    model = make_net()
    
    # 1. Grab a direct reference to the whitening layer (layer 0) BEFORE compiling.
    # We need this reference to toggle requires_grad inside the loop later.
    whitening_layer = model[0] 

    # 2. Initialize Whitening BEFORE compiling
    # (Accessing model[0] on a compiled model causes the 'not subscriptable' error)
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(whitening_layer, train_images)

    # 3. NOW we compile (The model structure is frozen, but weights can change)
    model = torch.compile(model)
    # --- FIX END ---

    current_steps = 0

    # Note: We must iterate over named_parameters of the ORIGINATING model if strictly needed, 
    # but usually compiled models support .named_parameters(). 
    # If this fails, we can use model._orig_mod.named_parameters()
    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.23)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (step - warmup_steps) / warmdown_steps
            return 1.0 * (1 - frac) + 0.07 * frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
    lookahead_state = LookaheadState(model)

    use_cuda_timing = (device.type == 'cuda')
    if use_cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    if use_cuda_timing:
        starter.record()
    else:
        start_time = time.time()

    # (Whitening init moved to top)

    if use_cuda_timing:
        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)
    else:
        total_time_seconds += time.time() - start_time

    for epoch in range(ceil(epochs)):
        
        # --- FIX: Use the saved reference, not model[0] ---
        whitening_layer.bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])
        
        if use_cuda_timing:
            starter.record()
        else:
            start_time = time.time()

        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_steps += 1
            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
            if current_steps >= total_train_steps:
                if lookahead_state is not None:
                    lookahead_state.update(model, decay=1.0)
                break

        if use_cuda_timing:
            ender.record()
            torch.cuda.synchronize()
            total_time_seconds += 1e-3 * starter.elapsed_time(ender)
        else:
            total_time_seconds += time.time() - start_time

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size
        val_acc = evaluate(model, test_loader, tta_level=0)
        run = None 

    if use_cuda_timing:
        starter.record()
    else:
        start_time = time.time()

    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])

    if use_cuda_timing:
        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)
    else:
        total_time_seconds += time.time() - start_time

    return tta_val_acc, total_time_seconds

# --- FIX 2: Correct Logging Logic ---
if __name__ == "__main__":
    # Ensure logs directory exists
    log_root = os.path.join('logs', EXPERIMENT_NAME)
    os.makedirs(log_root, exist_ok=True)
    
    # Correct path joining (Fixes Bug 2)
    csv_path = os.path.join(log_root, 'latest_run_log.csv')
    
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"\n>>> STARTING EXPERIMENT: {EXPERIMENT_NAME} <<<")
    print(f"    Batch Size: {BATCH_SIZE}, Width: {WIDTH_MULTIPLIER}x, Groups: {USE_GROUPED_CONV}")
    print_columns(logging_columns_list, is_head=True)
    
    with open(csv_path, 'w') as f:
        f.write('run,final_val_acc,total_time_seconds\n')

    results = []
    
    for run in range(N_RUNS):
        try:
            acc, seconds = main(run)
            results.append((acc, seconds))
            
            print(f"Run {run:03d} | Acc: {acc:.4f} | Time: {seconds:.3f}s")
            
            with open(csv_path, 'a') as f:
                f.write(f'{run},{acc},{seconds}\n')
                
        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user.")
            break
        except Exception as e:
            print(f"\n\nRun {run} failed with error: {e}")
            continue

    if len(results) > 0:
        results_tensor = torch.tensor(results)
        accs = results_tensor[:, 0]
        times = results_tensor[:, 1]

        print('\n' + '-'*30)
        print(f'Completed {len(results)} runs')
        print('Accuracy  - Mean: %.4f    Std: %.4f' % (accs.mean().item(), accs.std().item()))
        print('Time (s)  - Mean: %.4f    Std: %.4f' % (times.mean().item(), times.std().item()))
        print('-'*30)
        
        print(f"CSV log saved to: {os.path.abspath(csv_path)}")
        
        # Save .pt as well with correct path
        pt_path = os.path.join(log_root, f"final_results_{EXPERIMENT_NAME}.pt")
        torch.save({'accs': accs, 'times': times}, pt_path)
    else:
        print("No successful runs completed.")
