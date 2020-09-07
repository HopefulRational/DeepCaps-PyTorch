
# coding: utf-8

# In[1]:


'''
Authors: HopefulRational and team
'''

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
# import torch.autograd as grad
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import math
#import skimage.transform
#import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook as tqdm
#get_ipython().magic('matplotlib inline')

eps = 1e-8
cf = 1
# ONLY cuda runnable
device = torch.device("cuda:0")

# norm_squared = torch.sum(s**2, dim=dim, keepdim=True)
# return ((norm_squared /(1 + norm_squared + eps)) * (s / (torch.sqrt(norm_squared) + eps)))


# In[2]:


"""
From github: https://gist.github.com/ncullen93/425ca642955f73452ebc097b3b46c493
"""
"""
Affine transforms implemented on torch tensors, and
only requiring one interpolation
Included:
- Affine()
- AffineCompose()
- Rotation()
- Translation()
- Shear()
- Zoom()
- Flip()
"""

import math
import random
import torch

# necessary now, but should eventually not be
import scipy.ndimage as ndi
import numpy as np


def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is
    transformed about the center of the image. 
    NOTE: This is a fairly simple operaion, so can easily be
    moved to full torch.
    Arguments
    ---------
    matrix : 3x3 matrix/array
    x : integer
        height dimension of image to be transformed
    y : integer
        width dimension of image to be transformed
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='nearest', fill_value=0.):
    """Applies an affine transform to a 2D array, or to each channel of a 3D array.
    NOTE: this can and certainly should be moved to full torch operations.
    Arguments
    ---------
    x : np.ndarray
        array to transform. NOTE: array should be ordered CHW
    
    transform : 3x3 affine transform matrix
        matrix to apply
    """
    x = x.astype('float32')
    transform = transform_matrix_offset_center(transform, x.shape[1], x.shape[2])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
            final_offset, order=0, mode=fill_mode, cval=fill_value) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    return x

class Affine(object):

    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.):
        """Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.
        Arguments
        ---------
        rotation_range : one integer or float
            image will be rotated between (-degrees, degrees) degrees
        translation_range : a float or a tuple/list w/ 2 floats between [0, 1)
            first value:
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)
        shear_range : float
            radian bounds on the shear transform
        zoom_range : list/tuple with two floats between [0, infinity).
            first float should be less than the second
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
            ProTip : use 'nearest' for discrete images (e.g. segmentations)
                    and use 'constant' for continuous images
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        target_fill_mode : same as fill_mode, but for target image
        target_fill_value : same as fill_value, but for target image
        """
        self.transforms = []
        if translation_range:
            translation_tform = Translation(translation_range, lazy=True)
            self.transforms.append(translation_tform)
        
        if rotation_range:
            rotation_tform = Rotation(rotation_range, lazy=True)
            self.transforms.append(rotation_tform)

        if shear_range:
            shear_tform = Shear(shear_range, lazy=True)
            self.transforms.append(shear_tform) 

        if zoom_range:
            zoom_tform = Translation(zoom_range, lazy=True)
            self.transforms.append(zoom_tform)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value

    def __call__(self, x, y=None):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](x)
        for tform in self.transforms[1:]:
            tform_matrix = np.dot(tform_matrix, tform(x)) 

        x = torch.from_numpy(apply_transform(x.numpy(), tform_matrix,
            fill_mode=self.fill_mode, fill_value=self.fill_value))

        if y:
            y = torch.from_numpy(apply_transform(y.numpy(), tform_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
            return x, y
        else:
            return x

class AffineCompose(object):

    def __init__(self, 
                 transforms, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.):
        """Apply a collection of explicit affine transforms to an input image,
        and to a target image if necessary
        Arguments
        ---------
        transforms : list or tuple
            each element in the list/tuple should be an affine transform.
            currently supported transforms:
                - Rotation()
                - Translation()
                - Shear()
                - Zoom()
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        """
        self.transforms = transforms
        # set transforms to lazy so they only return the tform matrix
        for t in self.transforms:
            t.lazy = True
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value

    def __call__(self, x, y=None):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](x)
        for tform in self.transforms[1:]:
            tform_matrix = np.dot(tform_matrix, tform(x)) 

        x = torch.from_numpy(apply_transform(x.numpy(), tform_matrix,
            fill_mode=self.fill_mode, fill_value=self.fill_value))

        if y:
            y = torch.from_numpy(apply_transform(y.numpy(), tform_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
            return x, y
        else:
            return x


class Rotation(object):

    def __init__(self, 
                 rotation_range, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        degree = random.uniform(-self.rotation_range, self.rotation_range)
        theta = math.pi / 180 * degree
        rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                                    [math.sin(theta), math.cos(theta), 0],
                                    [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), rotation_matrix,
                fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), rotation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed


class Translation(object):

    def __init__(self, 
                 translation_range, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly translate an image some fraction of total height and/or
        some fraction of total width. If the image has multiple channels,
        the same translation will be applied to each channel.
        Arguments
        ---------
        translation_range : two floats between [0, 1) 
            first value:
                fractional bounds of total height to shift image
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                fractional bounds of total width to shift image 
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if isinstance(translation_range, float):
            translation_range = (translation_range, translation_range)
        self.height_range = translation_range[0]
        self.width_range = translation_range[1]
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        # height shift
        if self.height_range > 0:
            tx = random.uniform(-self.height_range, self.height_range) * x.size(1)
        else:
            tx = 0
        # width shift
        if self.width_range > 0:
            ty = random.uniform(-self.width_range, self.width_range) * x.size(2)
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.lazy:
            return translation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                translation_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), translation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed


class Shear(object):

    def __init__(self, 
                 shear_range, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly shear an image with radians (-shear_range, shear_range)
        Arguments
        ---------
        shear_range : float
            radian bounds on the shear transform
        
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        shear = random.uniform(-self.shear_range, self.shear_range)
        shear_matrix = np.array([[1, -math.sin(shear), 0],
                                 [0, math.cos(shear), 0],
                                 [0, 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                shear_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), shear_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed
      

class Zoom(object):

    def __init__(self, 
                 zoom_range, 
                 fill_mode='constant', 
                 fill_value=0, 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly zoom in and/or out on an image 
        Arguments
        ---------
        zoom_range : tuple or list with 2 values, both between (0, infinity)
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if self.lazy:
            return zoom_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                zoom_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), zoom_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed




# In[3]:


print("\nclass trans")
class trans(object):
    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.
                ):
       self.affine = Affine(rotation_range, translation_range, shear_range, zoom_range) 
    
    def __call__(self, data):
        data = transforms.ToTensor()(data)
        return self.affine(data)


# In[4]:


print("\nsquash -> Tensor")
print("softmax_3d -> Tensor")
print("one_hot -> numpy.array")

def squash(s, dim=-1):
    norm = torch.norm(s, dim=dim, keepdim=True)
    return (norm /(1 + norm**2 + eps)) * s

# not being used anymore. instead using nn.functional.softmax
def softmax_3d(x, dim):
  return (torch.exp(x) / torch.sum(torch.sum(torch.sum(torch.exp(x), dim=dim[0], keepdim=True), dim=dim[1], keepdim=True), dim=dim[2], keepdim=True))

def one_hot(tensor, num_classes=10):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor.cuda()) # One-hot encode
#     return torch.eye(num_classes).index_select(dim=0, index=tensor).numpy() # One-hot encode


# In[5]:


print("class ConvertToCaps")

class ConvertToCaps(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
      # channels first
      return torch.unsqueeze(inputs, 2)


# In[6]:


print("class FlattenCaps")

class FlattenCaps(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        # inputs.shape = (batch, channels, dimensions, height, width)
        batch, channels, dimensions, height, width = inputs.shape
        inputs = inputs.permute(0, 3, 4, 1, 2).contiguous()
        output_shape = (batch, channels * height * width, dimensions)
        return inputs.view(*output_shape)


# In[7]:


print("class CapsToScalars")

class CapsToScalars(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        # inputs.shape = (batch, num_capsules, dimensions)
        return torch.norm(inputs, dim=2)


# In[8]:


print("class Conv2DCaps")

# padding should be 'SAME'
# LATER correct: DONT PASS h, w FOR conv2d OPERATION

class Conv2DCaps(nn.Module):
    def __init__(self, h, w, ch_i, n_i, ch_j, n_j, kernel_size=3, stride=1, r_num=1):
        super().__init__()
        self.ch_i = ch_i
        self.n_i = n_i
        self.ch_j = ch_j
        self.n_j = n_j
        self.kernel_size = kernel_size
        self.stride = stride
        self.r_num = r_num
        in_channels = self.ch_i * self.n_i
        out_channels = self.ch_j * self.n_j
        self.pad = 1
        
#         self.w = nn.Parameter(torch.randn(ch_j, n_j, ch_i, n_i, kernel_size, kernel_size) * 0.01).cuda()
        
#         self.w_reshaped = self.w.view(ch_j*n_j, ch_i*n_i, kernel_size, kernel_size)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.pad).cuda()
        
        
    def forward(self, inputs):
        # check if happened properly
        # inputs.shape: (batch, channels, dimension, hight, width)
        
        self.batch, self.ch_i, self.n_i, self.h_i, self.w_i = inputs.shape
        in_size = self.h_i
        x = inputs.view(self.batch, self.ch_i * self.n_i, self.h_i, self.w_i)
        
        x = self.conv1(x)
        width = x.shape[2]
        x = x.view(inputs.shape[0], self.ch_j, self.n_j, width, width)
        return squash(x,dim=2)# squash(x).shape: (batch, channels, dimension, ht, wdth)


# In[9]:


print("class ConvCapsLayer3D")

# SEE kernel_initializer, 

class ConvCapsLayer3D(nn.Module):
  def __init__(self, ch_i, n_i, ch_j=32, n_j=4, kernel_size=3, r_num=3):
    
    super().__init__()
    self.ch_i = ch_i
    self.n_i = n_i
    self.ch_j = ch_j
    self.n_j = n_j
    self.kernel_size = kernel_size
    self.r_num = r_num
    in_channels = 1
    out_channels = self.ch_j * self.n_j
    stride = (n_i, 1, 1)
    pad = (0, 1, 1)
    
#     self.w = nn.Parameter(torch.randn(ch_j*n_j, 1, n_i, 3, 3)).cuda()
    
    
    self.conv1 = nn.Conv3d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=self.kernel_size,
                           stride=stride,
                           padding=pad).cuda()
    
    
  def forward(self, inputs):
    # x.shape = (batch, channels, dimension, height, width)
    self.batch, self.ch_i, self.n_i, self.h_i, self.w_i = inputs.shape
    in_size = self.h_i
    out_size = self.h_i

    x = inputs.view(self.batch, self.ch_i * self.n_i, self.h_i, self.w_i)
    x = x.unsqueeze(1)
    x = self.conv1(x)
    self.width = x.shape[-1]
    
    x = x.permute(0,2,1,3,4)
    x = x.view(self.batch, self.ch_i, self.ch_j, self.n_j, self.width, self.width)
    x = x.permute(0, 4, 5, 3, 2, 1).contiguous()
    self.B = x.new(x.shape[0], self.width, self.width, 1, self.ch_j, self.ch_i).zero_()
    x = self.update_routing(x, self.r_num)
    return x
  
  def update_routing(self, x, itr=3):
    # x.shape = (batch, width, width, n_j, ch_j, ch_i)    
    for i in range(itr):
      # softmax of self.B along (1,2,4)
      tmp = self.B.permute(0,5,3,1,2,4).contiguous().reshape(x.shape[0],self.ch_i,1,self.width*self.width*self.ch_j)
      #k = softmax_3d(self.B, (1,2,4))   # (batch, width, width, 1, ch_j, ch_i)
      #k = func.softmax(self.B, dim=4)
      k = func.softmax(tmp,dim=-1)
      k = k.reshape(x.shape[0],self.ch_i,1,self.width,self.width,self.ch_j).permute(0,3,4,2,5,1).contiguous()
      S_tmp = k * x
      S = torch.sum(S_tmp, dim=-1, keepdim=True)
      S_hat = squash(S)
      
      if i < (itr-1):
        agrements = (S_hat * x).sum(dim=3, keepdim=True)   # sum over n_j dimension
        self.B = self.B + agrements
      
    S_hat = S_hat.squeeze(-1)
    #batch, h_j, w_j, n_j, ch_j  = S_hat.shape
    return S_hat.permute(0, 4, 3, 1, 2).contiguous()


# In[10]:


print("class Mask_CID")

class Mask_CID(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x, target=None):
        # x.shape = (batch, classes, dim)
        # one-hot required
        if target is None:
            classes = torch.norm(x, dim=2)
            max_len_indices = classes.max(dim=1)[1].squeeze()
        else:
            max_len_indices = target.max(dim=1)[1]
        
#         print("max_len_indices: ", max_len_indices)
        increasing = torch.arange(start=0, end=x.shape[0]).cuda()
        m = torch.stack([increasing, max_len_indices], dim=1)
        
        masked = torch.zeros((x.shape[0], 1) + x.shape[2:])
        for i in increasing:
            masked[i] = x[m[i][0], m[i][1], :].unsqueeze(0)

        return masked.squeeze(-1), max_len_indices  # dim: (batch, 1, capsule_dim)


# In[11]:


print("class CapsuleLayer")

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules=10, num_routes=640, in_channels=8, out_channels=16, routing_iters=3): 
        # in_channels: input_dim;   out_channels: output_dim.
        super().__init__()
        
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.routing_iters = routing_iters
        
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels) * 0.01)
        self.bias = nn.Parameter(torch.rand(1, 1, num_capsules, out_channels) * 0.01)
    
    def forward(self, x):
        # x: [batch_size, 32, 16] -> [batch_size, 32, 1, 16]
        #                          -> [batch_size, 32, 1, 16, 1]
#         print("CapsuleLayer_x.shape: ", x.shape)
        x = x.unsqueeze(2).unsqueeze(dim=4)
        
        u_hat = torch.matmul(self.W, x).squeeze()  # u_hat -> [batch_size, 32, 10, 32]
        u_hat_detached = u_hat.detach() #detach the u_hat vector to stop the gradient flow during the calculation of the coefficients for dynamic routing.
        
        #   b_ij = torch.zeros((batch_size, self.num_routes, self.num_capsules, 1))
        b_ij = x.new(x.shape[0], self.num_routes, self.num_capsules, 1).zero_()
        
        for itr in range(self.routing_iters):
            c_ij = func.softmax(b_ij, dim=2)
            s_j  = (c_ij * u_hat_detached).sum(dim=1, keepdim=True) + self.bias #use detached u_hat during all the iteration except the final iteration.
            v_j  = squash(s_j, dim=-1)
            
            if itr < self.routing_iters-1:
                a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + a_ij
        v_j = v_j.squeeze() #.unsqueeze(-1)
        
        return v_j   # dim: (batch, num_capsules, out_channels or dim_capsules)


# In[12]:


print("class Decoder_mnist")

class Decoder_mnist(nn.Module):
    def __init__(self, caps_size=16, num_caps=1, img_size=28, img_channels=1):
        super().__init__()
        
        self.num_caps = num_caps
        self.img_channels = img_channels
        self.img_size = img_size

        self.dense = torch.nn.Linear(caps_size*num_caps, 7*7*16).cuda(device)
        self.relu = nn.ReLU(inplace=True)
                                     
        
        self.reconst_layers1 = nn.Sequential(nn.BatchNorm2d(num_features=16, momentum=0.8),
                                            
                                            nn.ConvTranspose2d(in_channels=16, out_channels=64, 
                                                               kernel_size=3, stride=1, padding=1
                                                              )
                                            )
        
        self.reconst_layers2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                                                  kernel_size=3, stride=2, padding=1
                                                 )
                               
        self.reconst_layers3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, 
                                                  kernel_size=3, stride=2, padding=1
                                                 )
                                            
        self.reconst_layers4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, 
                                                  kernel_size=3, stride=1, padding=1
                                                 )
                                            
        self.reconst_layers5 = nn.ReLU()
                                           
    
    
    def forward(self, x):
        # x.shape = (batch, 1, capsule_dim(=32 for MNIST))
        batch = x.shape[0]
        
        x = x.type(torch.FloatTensor)

        x = x.cuda()
        
        x = self.dense(x)
        x = self.relu(x)
        x = x.reshape(-1, 16, 7, 7)
        
        x = self.reconst_layers1(x)
        
        x = self.reconst_layers2(x)
        
        # padding
        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers3(x)

        # padding
        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers4(x)      
        
        x = self.reconst_layers5(x)
        x = x.reshape(-1, 1, self.img_size, self.img_size)
        return x  # dim: (batch, 1, imsize, imsize)


# In[13]:


print("class Model")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=128,
                                kernel_size=3, stride=1, padding=1)
        self.batchNorm = torch.nn.BatchNorm2d(num_features=128, eps=1e-08, momentum=0.99)
        self.toCaps = ConvertToCaps()
        
        self.conv2dCaps1_nj_4_strd_2 = Conv2DCaps(h=28, w=28, ch_i=128, n_i=1, ch_j=32, n_j=4, kernel_size=3, stride=2, r_num=1)
        self.conv2dCaps1_nj_4_strd_1_1 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps1_nj_4_strd_1_2 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps1_nj_4_strd_1_3 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=4, kernel_size=3, stride=1, r_num=1)
        
        self.conv2dCaps2_nj_8_strd_2 = Conv2DCaps(h=14, w=14, ch_i=32, n_i=4, ch_j=32, n_j=8, kernel_size=3, stride=2, r_num=1)
        self.conv2dCaps2_nj_8_strd_1_1 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps2_nj_8_strd_1_2 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps2_nj_8_strd_1_3 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        
        self.conv2dCaps3_nj_8_strd_2 = Conv2DCaps(h=7, w=7, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=2, r_num=1)
        self.conv2dCaps3_nj_8_strd_1_1 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps3_nj_8_strd_1_2 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps3_nj_8_strd_1_3 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        
        self.conv2dCaps4_nj_8_strd_2 = Conv2DCaps(h=4, w=4, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=2, r_num=1)
        self.conv3dCaps4_nj_8 = ConvCapsLayer3D(ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, r_num=3)
        self.conv2dCaps4_nj_8_strd_1_1 = Conv2DCaps(h=2, w=2, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
        self.conv2dCaps4_nj_8_strd_1_2 = Conv2DCaps(h=2, w=2, ch_i=32, n_i=8, ch_j=32, n_j=8, kernel_size=3, stride=1, r_num=1)
                
        self.decoder = Decoder_mnist(caps_size=16, num_caps=1, img_size=28, img_channels=1)
        self.flatCaps = FlattenCaps()
        self.digCaps = CapsuleLayer(num_capsules=10, num_routes=640, in_channels=8, out_channels=16, routing_iters=3)
        self.capsToScalars = CapsToScalars()
        self.mask = Mask_CID()
        self.mse_loss = nn.MSELoss(reduction="none")
    
    def forward(self, x, target=None):
        x = self.conv2d(x)
        x = self.batchNorm(x)
        x = self.toCaps(x)
        
        x = self.conv2dCaps1_nj_4_strd_2(x)
        x_skip = self.conv2dCaps1_nj_4_strd_1_1(x)
        x = self.conv2dCaps1_nj_4_strd_1_2(x)
        x = self.conv2dCaps1_nj_4_strd_1_3(x)
        x = x + x_skip
        
        x = self.conv2dCaps2_nj_8_strd_2(x)
        x_skip = self.conv2dCaps2_nj_8_strd_1_1(x)
        x = self.conv2dCaps2_nj_8_strd_1_2(x)
        x = self.conv2dCaps2_nj_8_strd_1_3(x)
        x = x + x_skip
        
        x = self.conv2dCaps3_nj_8_strd_2(x)
        x_skip = self.conv2dCaps3_nj_8_strd_1_1(x)
        x = self.conv2dCaps3_nj_8_strd_1_2(x)
        x = self.conv2dCaps3_nj_8_strd_1_3(x)
        x = x + x_skip
        x1 = x
        
        x = self.conv2dCaps4_nj_8_strd_2(x)
        x_skip = self.conv3dCaps4_nj_8(x)
        x = self.conv2dCaps4_nj_8_strd_1_1(x)
        x = self.conv2dCaps4_nj_8_strd_1_2(x)
        x = x + x_skip
        x2 = x
        
        xa = self.flatCaps(x1)
        xb = self.flatCaps(x2)
        x = torch.cat((xa, xb), dim=-2)
        dig_caps = self.digCaps(x)
        
        x = self.capsToScalars(dig_caps)
        masked, indices = self.mask(dig_caps, target)
        decoded = self.decoder(masked)

        return dig_caps, masked, decoded, indices
    
    def margin_loss(self, x, labels, lamda, m_plus, m_minus):
        v_c = torch.norm(x, dim=2, keepdim=True)
        tmp1 = func.relu(m_plus - v_c).view(x.shape[0], -1) ** 2
        tmp2 = func.relu(v_c - m_minus).view(x.shape[0], -1) ** 2
        loss = labels*tmp1 + lamda*(1-labels)*tmp2
        loss = loss.sum(dim=1)
        return loss
    
    def reconst_loss(self, recnstrcted, data):
        loss = self.mse_loss(recnstrcted.view(recnstrcted.shape[0], -1), data.view(recnstrcted.shape[0], -1))
        return 0.4 * loss.sum(dim=1)
    
    def loss(self, x, recnstrcted, data, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
        loss = self.margin_loss(x, labels, lamda, m_plus, m_minus) + self.reconst_loss(recnstrcted, data)
        return loss.mean()

        


# In[14]:


# loss
mse_loss = nn.MSELoss(reduction='none')

def margin_loss(x, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
        v_c = torch.norm(x, dim=2, keepdim=True)
        tmp1 = func.relu(m_plus - v_c).view(x.shape[0], -1) ** 2
        tmp2 = func.relu(v_c - m_minus).view(x.shape[0], -1) ** 2
        loss_ = labels*tmp1 + lamda*(1-labels)*tmp2
        loss_ = loss_.sum(dim=1)
        return loss_
    
def reconst_loss(recnstrcted, data):
        loss = mse_loss(recnstrcted.view(recnstrcted.shape[0], -1), data.view(recnstrcted.shape[0], -1))
        return 0.4 * loss.sum(dim=1)
    
def loss(x, recnstrcted, data, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
        loss_ = margin_loss(x, labels, lamda, m_plus, m_minus) + reconst_loss(recnstrcted, data)
        return loss_.mean()


# In[15]:


model = Model().cuda()
# lr = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# # torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
# lambda1 = lambda: epoch: lr * 0.5**(epoch // 10)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
batch_size = 64
num_epochs = 100
lamda = 0.5
m_plus = 0.9
m_minus = 0.1


# In[16]:


train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='/home/mtech3/CODES/ankit/data/FashionMNIST/FashionMNIST/',train=True,download=True,transform=trans(rotation_range=0.1, translation_range=0.1, zoom_range=(0.1, 0.2))),batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='/home/mtech3/CODES/ankit/data/FashionMNIST/FashionMNIST/',train=False,download=True,transform=transforms.ToTensor()),batch_size=batch_size,shuffle=True)


# In[17]:


def accuracy(indices, labels):
    correct = 0.0
    for i in range(indices.shape[0]):
        if float(indices[i]) == labels[i]:
            correct += 1
    return correct


# In[18]:


print("def test")

def test(model, test_loader, loss, batch_size, lamda=0.5, m_plus=0.9, m_minus=0.1):
  test_loss = 0.0
  correct = 0.0
  for batch_idx, (data, label) in enumerate(test_loader):
    data, labels = data.cuda(), one_hot(label.cuda())
    outputs, masked_output, recnstrcted, indices = model(data)
#     if batch_idx == 9:
#       print("test: ", indices)
    loss_test = model.loss(outputs, recnstrcted, data, labels, lamda, m_plus, m_minus)
    test_loss += loss_test.data
    indices_cpu, labels_cpu = indices.cpu(), label.cpu()
#     for i in range(indices_cpu.shape[0]):
#         if float(indices_cpu[i]) == labels_cpu[i]:
#             correct += 1
    correct += accuracy(indices_cpu, labels_cpu)
#     if batch_idx % 100 == 0:
#        print("batch: ", batch_idx, "accuracy: ", correct/len(test_loader.dataset))
#         print(indices_cpu)
  print("\nTest Loss: ", test_loss/len(test_loader.dataset), "; Test Accuracy: " , correct/len(test_loader.dataset) * 100,'\n')


# In[ ]:


def train(train_loader, model, num_epochs, lr=0.001, batch_size=64, lamda=0.5, m_plus=0.9,  m_minus=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    lambda1 = lambda epoch: 0.5**(epoch // 10)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.96)
    for epoch in range(num_epochs):
      for batch_idx, (data, label_) in enumerate(train_loader):
        data, label = data.cuda(), label_.cuda()
        labels = one_hot(label)
        optimizer.zero_grad()
        outputs, masked, recnstrcted, indices = model(data, labels)
        loss_val = model.loss(outputs, recnstrcted, data, labels, lamda, m_plus, m_minus)
        loss_val.backward()
        optimizer.step()
        if batch_idx%100 == 0:
          outputs, masked, recnstrcted, indices = model(data)
          loss_val = model.loss(outputs, recnstrcted, data, labels, lamda, m_plus, m_minus)
          print("epoch: ", epoch, "batch_idx: ", batch_idx, "loss: ", loss_val, "accuracy: ", accuracy(indices, label_.cpu())/indices.shape[0])
      test(model, test_loader, loss, batch_size, lamda, m_plus, m_minus)
      lr_scheduler.step()


# In[ ]:


# soft-training
train(train_loader, model, num_epochs=100, lr=0.001, batch_size=256, lamda=0.5, m_plus=0.9,  m_minus=0.1)

# Hard-Training
print("\n\n\n\nHard-Training\n")
train(train_loader, model, num_epochs=100, lr=0.001, batch_size=256, lamda=0.8, m_plus=0.95,  m_minus=0.05)

