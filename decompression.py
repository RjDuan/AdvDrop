# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# Local
import utils

def dequantize(image, q_table):
    """[summary]
    TODO: Add discription
    Args:
        image ([type]): [description]
        q_table ([type]): [description]

    Returns:
        [type]: [description]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image =  image.to(device)
    q_table = q_table.to(device)
    dequantitize_img = image * q_table 
    return dequantitize_img
    


def y_dequantize(image, y_table):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image =  image.to(device)
    y_table = y_table.to(device)
    # y_table = utils.y_table.to(device)
    dequantitize_img = image * (y_table )
    return dequantitize_img


def c_dequantize(image, c_table):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image =  image.to(device)
    c_table = c_table.to(device)
    # c_table = utils.c_table.to(device)
    dequantitize_img = image * (c_table )
    return dequantitize_img


def idct_8x8_ref(image):
    """ Reference Inverse Discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * alpha

    result = np.zeros((8, 8), dtype=np.float32)
    for u, v in itertools.product(range(8), range(8)):
        value = 0
        for x, y in itertools.product(range(8), range(8)):
            value += image[x, y] * np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
    result[u, v] = value
    return result * 0.25 + 128


def idct_8x8(image):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    # alpha = np.outer(alpha, alpha)
    alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float()).to(device)
    image = image.to(device)
    image = image * alpha
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)
    tensor =  nn.Parameter(torch.from_numpy(tensor).float()).to(device)
    result = 0.25 * torch.tensordot(image, tensor, dims=2) + 128
#    result = torch.from_numpy(result)
    result.view(image.shape)
    return result


def block_merging(patches, height, width):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """
    k = 8
    desired_height = int(np.ceil(height/k) * k)
    batch_size = patches.shape[0]
    image_reshaped = patches.view(batch_size, desired_height//k, desired_height//k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    image = image_transposed.contiguous().view(batch_size, desired_height, desired_height)
    image = image[:,:height, :width]
    return image


def chroma_upsampling(y, cb, cr):
    """ Upsample chroma layers
    Input: 
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """
    # def repeat(x, k=2):
    #     height, width = x.shape[1:3]
    #     x = x.unsqueeze(-1)
    #     x = x.repeat(1, 1, k, k)
    #     x = x.view(-1, height * k, width * k)
    #     return x

    # cb = repeat(cb)
    # cr = repeat(cr)

    # print(y.shape, cb.shape, cr.shape)
    return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


def ycbcr_to_rgb(image):
    """ Converts YCbCr image to RGB
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """
    matrix = np.array(
        [[298.082, 0, 408.583], [298.082, -100.291, -208.120],
         [298.082, 516.412, 0]],
        dtype=np.float32).T / 256
    shift = [-222.921, 135.576, -276.836]

    result = torch.tensordot(image, matrix, dims=1) + shift
    #result = torch.from_numpy(result)
    result.view(image.shape)
    return result.permute(0, 3, 1, 2)


def ycbcr_to_rgb_jpeg(image):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = np.array(
        [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
        dtype=np.float32).T
    shift = [0, -128, -128]
    image = image.to(device)
    shift = nn.Parameter(torch.tensor([0, -128., -128.])).to(device)
    matrix = nn.Parameter(torch.from_numpy(matrix)).to(device)
    result = torch.tensordot(image + shift, matrix, dims=1)
    #result = torch.from_numpy(result)
    result.view(image.shape)
    return result.permute(0, 3, 1, 2)


def decompress_jpeg(y, cb, cr, height, width, rounding=torch.round, factor=1):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """
    upresults = {}
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = c_dequantize(components[k], factor) if k in (
            'cb', 'cr') else y_dequantize(components[k], factor)
        comp = idct_8x8(comp)
        comp = block_merging(comp, int(height), int(width)
                             ) if k in ('cb', 'cr') else block_merging(comp, height, width)
        upresults[k] = comp
    image = chroma_upsampling(upresults['y'], upresults['cb'], upresults['cr'])
    # print(image)
    image = ycbcr_to_rgb_jpeg(image)
    return image
