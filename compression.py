# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# Local
import utils
from utils import * 
from decompression import *
from PIL import Image

# pytorch models
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torchvision.transforms as transforms

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g 
    
def sgn( x):
    x = RoundWithGradient.apply(x)
    return x

def rgb_to_ycbcr(image):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = np.array(
        [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.],
         [112., -93.786, -18.214]],
        dtype=np.float32).T 
    shift = [16., 128., 128.]
    image = image
    image = image.permute(0, 2, 3, 1)
    result = torch.tensordot(image, torch.from_numpy(matrix).to(device), dims=1) + shift
#    result = torch.from_numpy(result)
    result.view(image.shape)
    return result


def rgb_to_ycbcr_jpeg(image):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=np.float32).T
    shift =  nn.Parameter(torch.tensor([0., 128., 128.])).to(device)
    image = image.permute(0, 2, 3, 1)
    result = torch.tensordot(image, torch.from_numpy(matrix).to(device), dims=1) + shift
#    result = torch.from_numpy(result)
    result.view(image.shape)
    return result


def chroma_subsampling(image):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """
    # image_2 = image.permute(0, 3, 1, 2).clone()
    # avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
    #                         count_include_pad=False)
    # cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
    # cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
    # cb = cb.permute(0, 2, 3, 1)
    # cr = cr.permute(0, 2, 3, 1)
    #return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)
    return image[:, :, :, 0], image[:, :, :, 1], image[:, :, :, 2]


def block_splitting(image, block_size = 8):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    TODO:
    1. Incorrect when batch size> 1 
    """
    k = block_size
    height, width = image.shape[1:3]
    desired_height = int(np.ceil(height/k) * k)
    batch_size = image.shape[0]
    desired_img = torch.zeros(batch_size, desired_height, desired_height)
    desired_img[:,:height, :width] = image
    image_reshaped = desired_img.view(batch_size, desired_height // k, k, -1, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(batch_size, -1, k, k)


def dct_8x8_ref(image):
    """ Reference Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    image = image - 128
    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2 * x + 1) * u *
                                          np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        result[u, v] = value
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    return result * scale


def dct_8x8(image):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    tensor =  nn.Parameter(torch.from_numpy(tensor).float())
    scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float() )
    result = scale * torch.tensordot(image, tensor, dims=2)
    #result = torch.from_numpy(result)
    result.view(image.shape)
    return result

def quantize(image, q_table,alpha):
    """[summary]
    TODO: add disciption.

    Args:
        image ([type]): [description]
        q_table ([type]): [description]
    """
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    q_table = q_table.to(device)
    pre_img = image/(q_table)
    after_img = phi_diff(pre_img, alpha)
    # after_img = sgn(after_img)
    # after_img = torch.round(pre_img) + torch.empty_like(pre_img).uniform_(0.0, 1.0)
    # diff = after_img - pre_img
    # print("Max difference: ", torch.max(diff))
    # image = torch.round(image)
    # image = diff_round(image)
    # after_img = diff_round(pre_img)
    return after_img

def y_quantize(image, y_table):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    y_table = y_table.to(device)
    image = image / (y_table)
    image = torch.round(image)
    return image


def c_quantize(image, c_table):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
        TODO:
        1. c_table global
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    c_table = c_table.to(device)
    # c_table = utils.c_table.to(device)
    image = image / (c_table)
    image = torch.round(image)
    return image


def compress_jpeg(imgs, rounding=torch.round, factor=1):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """
    temp = rgb_to_ycbcr_jpeg(imgs)
    y, cb, cr = chroma_subsampling(temp)
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = block_splitting(components[k])
        comp = dct_8x8(comp)
        comp = c_quantize(comp, torch.round, factor=factor) if k in (
            'cb', 'cr') else y_quantize(comp, torch.round, factor=factor)

        components[k] = comp
    return components['y'], components['cb'], components['cr']

def CWLoss(logits, target = 900, kappa=0):
    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target = torch.ones(logits.size(0)).type(logits.type()).fill_(target)
    target_one_hot = torch.eye(1000).type(logits.type())[target.long()]
    print(target_one_hot.shape)
    # workaround here.
    # subtract large value from target class to find other max value
    # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    return torch.sum(torch.max(other-real, kappa))

if __name__ == "__main__":
    
    # Data trnasform
    preprocess = transforms.Compose(
        [transforms.Resize(256),
        #  transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the Resnet50
    model = resnet50(pretrained=True)
    model.eval()
    
    img_path = './data/imagenet/giant_panda/1.jpg'
    img_320 = np.asarray(Image.open(img_path).convert("RGB").resize((320,320)), dtype = np.float32)
    w, h, _ = img_320.shape
    # print(img.shape)
    
    img =  np.moveaxis(img_320, 2, 0)
    batch_img = torch.from_numpy(np.expand_dims(img, axis = 0))
    print(batch_img.shape)
    batch_img.to(device)
    with open('./imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    for quality in range(100, 0, -10):
        fa = utils.quality_to_factor(quality)
        print("Quality: ", quality)
        print("fa: ", fa)
        # y, cb, cr = compress_jpeg(batch_img, rounding=torch.round, factor=fa)
        # # print(test_com[0].shape)
        # # print(test_com[1].shape)
        # # print(test_com[2].shape)
        # result_img = decompress_jpeg(y, cb, cr, 320, 320, rounding=torch.round, factor=fa)
        # # print(result_img)
        # img_np = result_img.cpu().detach().numpy()
        # print(img_np.shape)
        # img_np = img_np[0]
        # img_np = np.moveaxis(img_np, 0, 2)
        # print(img_np.shape)
        
        # input_pil = Image.fromarray(np.uint8(img_np))        
        # input_tensor = preprocess(input_pil)
        # input_batch = input_tensor.unsqueeze(0)


        # if torch.cuda.is_available():
        #     input_batch = input_batch.to('cuda')
        #     model.to('cuda')
    
        # out = model(input_batch)

        # # print(output[0])
        # # print(torch.nn.functional.softmax(output[0], dim = 0))
        # _, index = torch.max(out, 1)

        # percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        # print(labels[index[0]], percentage[index[0]].item())
        
        # _, indices = torch.sort(out, descending=True)
        # results = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
        # print(results)
    
        
        # img_np = img_np[0]
        # print(img_np)
        # compressed_img = np.moveaxis(img_np, 0, 2)
        # saved_img = Image.fromarray(np.uint8(compressed_img))
        # saved_img.save('./test_imgs/1_compressed_factor_'+ str(fa) + '.png', "png")
        
        # org_img = Image.fromarray(np.uint8(img_320))
        # org_img.save('./test_imgs/1_org_320.png', "png")
        # jpeg_compressor = DiffJPEG(w, h, True, 100)
        # x = jpeg_compressor(batch_img)
        # x_np = x.cpu().detach().numpy()
        # print(x_np.shape)
        # compressed_img = x_np[0]
        # compressed_img = np.moveaxis(compressed_img, 0, 2)
        # print("Max value:", np.amax(compressed_img))
        # saved_img = Image.fromarray(np.uint8(compressed_img))
        # saved_img.save('./test_imgs/1_compressed.jpg', "JPEG")