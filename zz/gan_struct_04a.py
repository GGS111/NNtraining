#!pip3 install torch==1.5.0 !pip3 install torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html 
import torch

from torch import nn
from torch.nn import Conv2d
from torch.nn import LeakyReLU
#from torch.nn import ReLU
from torch.nn import Linear
from torch.nn import Softmax
from torch.nn import MaxPool2d
#from torch.nn import Dropout
from torch.nn import Sigmoid
from torch.nn import BatchNorm2d,AvgPool2d
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Flatten
from torch.nn import LeakyReLU
from torch.nn import ReLU
from torch.nn import Dropout
from torch.nn import BatchNorm1d
from torch.nn import Conv2d
from torch.nn import ConvTranspose2d
from torch.nn import Softmax  
from torch.nn import MaxPool2d,AvgPool2d
from enum import Enum
from torch.nn import Conv2d
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from .layers import Layer
from .layers.Input import Input
from .layers.Lambda import Lambda
from .layers.Reshape import Reshape
#from .layers.Flatten import Flatten
from .utils.torchsummary import summary as _summary
from .utils.WrappedDataLoader import WrappedDataLoader
from .utils.History import History
from .utils.Regularizer import Regularizer
from .layers.Layer_04_uno import Layer_04_uno
from .layers.Layer_01 import Layer_01 
 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
#from .Conv2D_SA_onnx import Conv2D_SA_onnx
from .Convolution_uno_01   import conv_layer_universal_uno_04,conv_layer_universal_uno_05
import numpy as np
from enum import Enum
 

import torchvision 
###########################################
txt_00_f='./txt_00.JPG'
#############################################################################3
###########################################################################
class fully_conn_layer_universal_00(Layer_01):
    def __init__(self, Size_, last_activate, device = None):
        super(fully_conn_layer_universal_00, self).__init__()
        self.Size = Size_[0]
        self.last_activate=last_activate
        self.class_name = self.__class__.__name__
        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _layer_D00 = Linear( self.Size,  Size_[1], bias = True)
        self.add_module('D00', _layer_D00) 
        _layer_Dropout00 = Dropout(0.2)
        self.add_module('Dropout00', _layer_Dropout00) 
        
        _layer_activation_LW1 = LeakyReLU(0.2) 
        self.add_module('activation_LW1', _layer_activation_LW1) 
        
        _layer_D01 = Linear(Size_[1], Size_[2], bias = True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.2)
        self.add_module('Dropout01', _layer_Dropout01) 
        
        
        
        _layer_activation_LW2 = LeakyReLU(0.05) 
        self.add_module('activation_LW2', _layer_activation_LW2) 
        _layer_Dropout02 = Dropout(0.1)
        self.add_module('Dropout02', _layer_Dropout02) 
        
        _layer_D02 = Linear(Size_[2], Size_[3], bias = True)
        self.add_module('D02', _layer_D02)

        _layer_activation_LW3 = LeakyReLU(0.05) 
        self.add_module('activation_LW3', _layer_activation_LW3) 
        
        _layer_D03 = Linear(Size_[3], Size_[4], bias = True)
        self.add_module('D03', _layer_D03)
        
        _layer_activation_LW4 = LeakyReLU(0.05) 
        self.add_module('activation_LW4', _layer_activation_LW4) 
        
        _layer_D04 = Linear(Size_[4], Size_[-1], bias = True)
        self.add_module('D04', _layer_D04)
        
        
        
        if last_activate == 'sigmoid':
            _layer_activation_D4 = Sigmoid()
            self.add_module('activation_last', _layer_activation_D4)
        self.to(self.device)
        self.reset_parameters()
        
    def forward(self, x):
         
         
        
        l1=self._modules['D00'](x)
        l1=self._modules['Dropout00'](l1)
        dense_01=self._modules['activation_LW1'](l1)
        l2=self._modules['D01'](dense_01)
        l2=self._modules['Dropout01'](l2)
        dense_02=self._modules['activation_LW2'](l2)
        l3=self._modules['D02'](dense_02)
        l3=self._modules['Dropout02'](l3)
        dense_03=self._modules['activation_LW3'](l3)
        l4=self._modules['D03'](dense_03)
        dense_04=self._modules['activation_LW4'](l4)
        l5=self._modules['D04'](dense_04)
        if self.last_activate == 'sigmoid':
            l5 = self._modules['activation_last'](l5) 
             
        y=l5    
        y = self._contiguous(y)
        return y
################################################3

class conv_layer_downsample_01(Layer_01):
    def __init__(self, numfilters1_in,   numfilters1_out,  bias_, L1 = 0., L2 = 0.,device = None):
        super(conv_layer_downsample_01, self).__init__()
        
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.class_name = self.__class__.__name__
        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

#        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
#        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
#        
#        self.add_module('permut_channelfirst', _layer_permut_channelfirst)
#        self.add_module('permut_channellast', _layer_permut_channellast)
        
        ###################################################
        _layer_conv_31 = Conv2d(numfilters1_in,numfilters1_out, kernel_size=(3, 3),
                            stride=(1, 1), padding = (1, 1), padding_mode = 'zeros', bias = bias_)
        
         

        _layer_batch_norm_1 = BatchNorm2d(num_features=numfilters1_out)#, affine=False)
         
        
        _layer_pooling_1 = AvgPool2d(kernel_size=(2, 2))            
        _layer_activation_1 = LeakyReLU(0.05) 
        ###############
        
        self.add_module('conv_31', _layer_conv_31)
         
        self.add_module('batch_norm_1', _layer_batch_norm_1)
         
        self.add_module('pooling_1', _layer_pooling_1)
        self.add_module('activation_1', _layer_activation_1)
        
        self.to(self.device)
        self.reset_parameters()
        
         
    def forward(self,img_23_32_64_32 ):
         
        img_31 = self._call_simple_layer('conv_31', img_23_32_64_32)
        img_32 = self._call_simple_layer('batch_norm_1', img_31)
        img_33 = self._call_simple_layer('activation_1', img_32)
        img_34_16_32_64 = self._call_simple_layer('pooling_1', img_33)

        

        return img_34_16_32_64
    
############################################################################################
class Layer_06(torch.nn.Module):
    def __init__(self, *input_shapes , **kwargs):
        super(Layer_06, self).__init__(**kwargs )
        self.input_shapes = input_shapes
        self.eps_=10**(-20)
        self._criterion = None
        self._optimizer = None
        
    def reset_parameters(self):
        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            return (-lim, lim)
        
        for module in self._modules.values():
            if hasattr(module, 'weight') and (module.weight is not None):
                module.weight.data.uniform_(*hidden_init(module))
            if hasattr(module, 'bias') and (module.bias is not None):
                module.bias.data.fill_(0)

    def _get_regularizer(self):
        raise Exception("Need to override method _get_regularizer()!");
        
    def summary(self):
        _summary(self, input_size = self.input_shapes, device = self.device)

    def weights_is_nan(self):
        is_nan = False
        for module in self._modules.values():
            if hasattr(module, 'weight'):
                if ((isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach())) or
                     (isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach()))):
                    is_nan = True
                    break
            if hasattr(module, 'bias'):
                if (isinstance(module.bias, torch.Tensor) and torch.isnan(torch.sum(module.bias.data).detach())):
                    is_nan = True
                    break
            
        return is_nan

    def save_state(self, file_path):
        torch.save(self.state_dict(), file_path)
        
    def load_state(self, file_path):
        try:
            print()
            print('Loading preset weights... ', end='')

            self.load_state_dict(torch.load(file_path))
            self.eval()
            is_nan = False
            for module in self._modules.values():
                if hasattr(module, 'weight'):
                    if ((isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach())) or
                         (isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach()))):
                        is_nan = True
                        break
                if hasattr(module, 'bias'):
                    if (isinstance(module.bias, torch.Tensor) and torch.isnan(torch.sum(module.bias.data).detach())):
                        is_nan = True
                        break
                    
            if (is_nan):
                raise Exception("[Error]: Parameters of layers is NAN!")
                
            print("Ok.")
        except Exception as e:
            print("Fail! ", end='')
            print(str(e))
            print("[Action]: Reseting to random values!")
            self.reset_parameters()
            
    def cross_entropy_00(self, pred, soft_targets):
        return -torch.log(self.eps_+torch.mean(torch.sum(soft_targets * pred, -1)))
    
    def MSE_00(self, pred, soft_targets):
        return   torch.mean(torch.mean((soft_targets - pred)**2, -1)) 

    def compile(self, criterion, optimizer,   **kwargs):
        
        if criterion == 'mse-mean':
            self._criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mse-sum':
            self._criterion = nn.MSELoss(reduction='sum')
        elif criterion == '000':
            self._criterion = self.MSE_00 
        elif criterion == '001':
            self._criterion = self.cross_entropy_00

        else:
            raise Exception("Unknown loss-function!")
            
        if (optimizer == 'sgd'):
             
            momentum = 0.2
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            if ('momentum' in kwargs.keys()):
                momentum = kwargs['momentum']
            self._optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum);
        elif (optimizer == 'adam'):
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            self._optimizer   = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False) 
             
        else:
            raise Exception("Unknown optimizer!")
            
 
    def _call_simple_layer(self, name_layer, x):
        y = self._modules[name_layer](x)
        if self.device.type == 'cuda' and not y.is_contiguous():
            y = y.contiguous()
        return y
    
    def _contiguous(self, x):
        if self.device.type == 'cuda' and not x.is_contiguous():
            x = x.contiguous()
        return x
#####################################################################3

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№ 
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№ 
class Layer_09(nn.Module):
    def __init__(self, *input_shapes ,  **kwargs):
        super(Layer_09, self).__init__(**kwargs )
         
        self.input_shapes = input_shapes
        self.eps_=10**(-20)
        self._criterion = None
        self._optimizer = None
        self.transforms_ = 0#torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2.0))
         

        
    def reset_parameters(self):
        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            return (-lim, lim)
        
        for module in self._modules.values():
            if hasattr(module, 'weight') and (module.weight is not None):
                module.weight.data.uniform_(*hidden_init(module))
            if hasattr(module, 'bias') and (module.bias is not None):
                module.bias.data.fill_(0)

    def _get_regularizer(self):
        raise Exception("Need to override method _get_regularizer()!");
        
    def summary(self):
        _summary(self, input_size = self.input_shapes, device = self.device)

    def weights_is_nan(self):
        is_nan = False
        for module in self._modules.values():
            if hasattr(module, 'weight'):
                if ((isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach())) or
                     (isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach()))):
                    is_nan = True
                    break
            if hasattr(module, 'bias'):
                if (isinstance(module.bias, torch.Tensor) and torch.isnan(torch.sum(module.bias.data).detach())):
                    is_nan = True
                    break
            
        return is_nan

    def save_state(self, file_path):
         
        torch.save(self.state_dict(), file_path)
        
    def load_state(self, file_path):
        try:
            print()
            print('Loading preset weights... ', end='')

            self.load_state_dict(torch.load(file_path))
            self.eval()
            is_nan = False
            for module in self._modules.values():
                if hasattr(module, 'weight'):
                    if ((isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach())) or
                         (isinstance(module.weight, torch.Tensor) and torch.isnan(torch.sum(module.weight.data).detach()))):
                        is_nan = True
                        break
                if hasattr(module, 'bias'):
                    if (isinstance(module.bias, torch.Tensor) and torch.isnan(torch.sum(module.bias.data).detach())):
                        is_nan = True
                        break
                    
            if (is_nan):
                raise Exception("[Error]: Parameters of layers is NAN!")
                
            print("Ok.")
        except Exception as e:
            print("Fail! ", end='')
            print(str(e))
            print("[Action]: Reseting to random values!")
            self.reset_parameters()
            
    def cross_entropy_00(self, pred, soft_targets):
        return -torch.log(self.eps_+torch.mean(torch.sum(soft_targets * pred, -1)))
    
    def MSE_00(self, pred, soft_targets):
        return   torch.mean(torch.mean((soft_targets - pred)**2, -1)) 
    def MSE_00a(self, pred, soft_targets):
        out_pred=torch.mean(pred,-1)
        out_true=torch.mean(soft_targets,-1)
        MSELoss=nn.MSELoss(reduction='mean')
        return MSELoss(out_pred, out_true)
    
    def MSE_00b(self, pred, soft_targets):
        out_pred=torch.mean(self.transforms_(pred),-1)
        out_true=torch.mean(self.transforms_(soft_targets),-1)
        MSELoss=nn.MSELoss(reduction='mean')
        return MSELoss(out_pred, out_true)
        
        
    
   
    def MSE_01(self, pred, soft_targets):
        diff=(soft_targets - pred)[...,0]
        
        diff_L=self.transforms_(diff)
        diff_H=diff-diff_L
        L_H=torch.mean(diff_H**2) 
        L_L=torch.mean(diff_L**2) 
        if 0:
            #print((soft_targets - pred).shape)
            print(L_L,L_H )
            diff_L_=diff_L.detach().cpu().numpy()[0,:,:]
            diff_H_=diff_H.detach().cpu().numpy() [0,:,:]
            plot_im_2(diff_L_[::13,::17],diff_H_[::13,::17])

            #ai_2(diff_L.detach().cpu().numpy()[0,:,:])
            #ai_2(diff_H.detach().cpu().numpy() [0,:,:])
        return   1.0*L_L+1000.0*L_H
    
    
    def vgg_loss_func(self,ypred, ytrue):
        out_pred = self.vgg_conv(ypred)
        out_true = self.vgg_conv(ytrue)
        MSELoss=nn.MSELoss(reduction='mean')
        return MSELoss(out_pred, out_true)

    def MSE_01b(self, pred, soft_targets):
        #print('self.device',self.device)
        diff=(soft_targets - pred) 
        
        diff_L=self.transforms_(diff)
        diff_H=diff-diff_L
        L_H=torch.mean(diff_H**2) 
        L_L=torch.mean(diff_L**2) 
        if 0:
            #print((soft_targets - pred).shape)
            print(L_L,L_H )
            diff_L_=diff_L.detach().cpu().numpy()[0,:,:]
            diff_H_=diff_H.detach().cpu().numpy() [0,:,:]
            plot_im_2(diff_L_[::13,::17],diff_H_[::13,::17])

            #ai_2(diff_L.detach().cpu().numpy()[0,:,:])
            #ai_2(diff_H.detach().cpu().numpy() [0,:,:])
        return   1.0*L_L+0.000000001*L_H
    
    
    def MSE_01a(self, pred, soft_targets):
        #print('self.device',self.device)
        diff=(soft_targets - pred)[...,0]
        
        diff_L=self.transforms_(diff)
        diff_H=diff-diff_L
        L_H=torch.mean(diff_H**2) 
        L_L=torch.mean(diff_L**2) 
        if 0:
            #print((soft_targets - pred).shape)
            print(L_L,L_H )
            diff_L_=diff_L.detach().cpu().numpy()[0,:,:]
            diff_H_=diff_H.detach().cpu().numpy() [0,:,:]
            plot_im_2(diff_L_[::13,::17],diff_H_[::13,::17])

            #ai_2(diff_L.detach().cpu().numpy()[0,:,:])
            #ai_2(diff_H.detach().cpu().numpy() [0,:,:])
        return   1.0*L_L+0.000000001*L_H
    
    def MSE_02(self, pred, soft_targets):
        soft_targets_=soft_targets[...,0]
        pred_=pred[...,0]
        txt_L=self.transforms_(soft_targets_)
        txt_H=soft_targets_-txt_L
        
        diff_L=(txt_L - pred_) 
        
         
        diff_H=(txt_H - pred_) 
        L_H=torch.mean(diff_H**2) 
        L_L=torch.mean(diff_L**2) 
        if 0:
            #print((soft_targets - pred).shape)
            print(L_L,L_H )
            diff_L_=diff_L.detach().cpu().numpy()[0,:,:]
            diff_H_=diff_H.detach().cpu().numpy() [0,:,:]
            plot_im_2(diff_L_[::13,::17],diff_H_[::13,::17])

            ai_2(txt_L.detach().cpu().numpy()[0,:,:])
            ai_2(txt_H.detach().cpu().numpy() [0,:,:])
        return   1.0*L_L+10.0*L_H
    
    def MSE_02a(self, pred, soft_targets):
        soft_targets_=soft_targets[...,0]
        pred_=pred[...,0]
        txt_L=self.transforms_(soft_targets_)
        txt_H=soft_targets_-txt_L
        pred_L=self.transforms_(pred_)
        pred_H=pred_-pred_L
        #ai_2(pred_H.detach().cpu().numpy()[0,:,:])
        l1=  torch.mean(pred_H**2 )  
        MSELoss=nn.MSELoss(reduction='mean')
        l2=MSELoss(txt_L, pred_L)
        l3=MSELoss(txt_H, pred_H)
         
        return   2.1*(l2+2*l3)+(0.5-l1)
   
    
    def MSE_03(self, pred, soft_targets):
        l1= 1*torch.mean(torch.mean((soft_targets[:,:,:,0] - pred[:,:,:,0])**2, -1))
        l2= 25*torch.mean(torch.mean((soft_targets[:,:,:,1:] - pred[:,:,:,1:])**2, -1)) 
        return   l1+ l2
    def MSE_04(self, pred, soft_targets):
        a=soft_targets[:,:,:,0] - pred[:,:,:,0]
        
        l1=  torch.abs(a)**4 
        
        #print(torch.mean(l1)) 
        return     torch.mean(l1) 
    
    def TL_00(self, pred,true ):
        positive=pred[:,0].reshape([ pred.shape[0],1])
        negative=pred[:,1].reshape([ pred.shape[0],1])
        #plot_im_2(pred[:,0].cpu().detach().numpy(),pred[:,1].cpu().detach().numpy())
        alpha_= 1.0  
        loss_vec= negative -positive + alpha_
             
        
        TLoss= torch.mean(torch.mean(loss_vec, -1)) 
         
        return TLoss

    def compile(self, criterion, optimizer, **kwargs):
         
        if criterion == '010':             
            self._criterion = nn.MSELoss(reduction='mean')
        elif criterion == '010a':             
            self._criterion = self.MSE_00a
        elif criterion == '010b':             
            self._criterion = self.MSE_00b

        elif criterion == '011':             
            self._criterion = self.MSE_01 
        elif criterion == '011a':             
            self._criterion = self.MSE_01a 
        elif criterion == '011b':             
            self._criterion = self.MSE_01b 

        elif criterion == '012':             
            self._criterion = self.MSE_02 
        elif criterion == '012a':             
            self._criterion = self.MSE_02a 

        elif criterion == '013':
            self._criterion = self.MSE_03 
        elif criterion == '014':
            self._criterion = self.MSE_04 

        elif criterion == 'mse-sum':
            self._criterion = nn.MSELoss(reduction='sum')
        elif criterion == '000':
            self._criterion = self.MSE_00 
        elif criterion == '001':
            self._criterion = self.cross_entropy_00
        elif criterion == 'TL00':
            self._criterion = self.TL_00
        elif criterion == 'vgg_loss':
            self._criterion = self.vgg_loss_func

 
        else:
            print('criterion',criterion)
            raise Exception("Unknown loss-function!")
            
        if (optimizer == 'sgd'):
            lr = 0.001
            momentum = 0.2
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            if ('momentum' in kwargs.keys()):
                momentum = kwargs['momentum']
            self._optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum);
        elif (optimizer == 'adam'):
            lr = 0.001
            betas = (0.9, 0.999)
            eps = 1e-8
            if ('lr' in kwargs.keys()):
                lr = kwargs['lr']
            if ('betas' in kwargs.keys()):
                momentum = kwargs['betas']
            if ('eps' in kwargs.keys()):
                eps = kwargs['eps']

            self._optimizer = optim.Adam(self.parameters(), lr=lr, betas=betas, eps=eps);
        else:
            raise Exception("Unknown optimizer!")
            
 
    def _call_simple_layer(self, name_layer, x):
        y = self._modules[name_layer](x)
        if self.device.type == 'cuda' and not y.is_contiguous():
            y = y.contiguous()
        return y
    
    def _contiguous(self, x):
        if self.device.type == 'cuda' and not x.is_contiguous():
            x = x.contiguous()
        return x

##########################################################################3
#####################################################################3
class conv_layer_universal_upsample_05( Layer):
    def __init__(self, numfilters_in, numfilters_out, k_size,bias_, L1 = 0., L2 = 0., device = None):
        super(conv_layer_universal_upsample_05, self).__init__()
        
        self.class_name = self.__class__.__name__
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _layer_deconv_01 =  ConvTranspose2d(numfilters_in,numfilters_out, kernel_size=(k_size, k_size), 
                                stride=(2, 2), padding = (1, 1),padding_mode = 'zeros', output_padding=(1, 1), bias = True)
        self.add_module('deconv_01', _layer_deconv_01 )
        _layer_activation_D0 = LeakyReLU(0.05) 
        self.add_module('activation_D0', _layer_activation_D0) 
        _layer_conv_4 = Conv2d(numfilters_out, numfilters_out, kernel_size=(k_size, k_size),
                     stride=(1, 1), padding=(1, 1), padding_mode = 'zeros', bias = True)
        self.add_module('conv_4', _layer_conv_4 )
        _layer_batch_norm_1 = BatchNorm2d(numfilters_out)
        self.add_module('batch_norm_1', _layer_batch_norm_1)
        _layer_activation_D1 = LeakyReLU(0.05)
        self.add_module('activation_D1', _layer_activation_D1) 
        self.to(self.device)
        #self.reset_parameters()
        
    def forward(self, img_23_32_64_32):
        img_31 = self._modules['deconv_01'](img_23_32_64_32)
        img_32 = self._modules['activation_D0'](img_31)
        img_33 = self._modules['conv_4'](img_32)
        img_33 = self._modules['batch_norm_1'](img_33)
        img_34 = self._modules['activation_D1'](img_33)
        
        return img_34

    
#############################################################################################3
################################################3
################################################3
class quasy_conv_00(Layer_01):
    def __init__( self , n_size=11, canal_in_0=3, cannal_out_0=5,last_act='relu',\
                 L1 = 0., L2 = 0.,device = None,show=0):
        super( quasy_conv_00 , self).__init__()         
        self.class_name = self.__class__.__name__        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.L1=L1
        self.L2=L2
        self.add_module('conv_layer_universal_uno_100', conv_layer_universal_uno_05\
                        (canal_in_0, n_size**2, 1,'relu',n_size,int(n_size/2),1, self.device)
                       )
        self.add_module('conv_layer_universal_uno_101', conv_layer_universal_uno_05\
                        (n_size**2, int(n_size**2/3), 0,'relu',1,0,1, self.device)
                       )
        self.add_module('conv_layer_universal_uno_102', conv_layer_universal_uno_05\
                        (int(n_size**2/3), int(n_size**2/5), 0,'relu',1,0,1, self.device)
                       )
        self.add_module('conv_layer_universal_uno_103', conv_layer_universal_uno_05\
                        (int(n_size**2/5), int(n_size**2/7), 0,'relu',1,0,1, self.device)
                       )
        self.add_module('conv_layer_universal_uno_104', conv_layer_universal_uno_05\
                        (int(n_size**2/7), cannal_out_0, 0,last_act,1,0,1, self.device)
                       )

        
        
        self.to(self.device)
        self.reset_parameters()
        
    ###############################################################        
    def forward(self, im):
         
         
         
         
        im_100=self._modules['conv_layer_universal_uno_100'](im)         
        im_101=self._modules['conv_layer_universal_uno_101'](im_100)
        im_102=self._modules['conv_layer_universal_uno_102'](im_101)
        im_103=self._modules['conv_layer_universal_uno_103'](im_102)
        im_104=self._modules['conv_layer_universal_uno_104'](im_103)
         

        
        y = self._contiguous(im_104)
        return y
################################################3
class Normalization(nn.Module):
    def __init__(self,  device):
        super(Normalization, self).__init__()
        self.device =device
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]) 
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]) 
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.FloatTensor(cnn_normalization_mean).view(-1, 1, 1).to(self.device)
        #self.mean = torch.tensor(mean)
        #print(self.mean)
        #self.std = torch.tensor(std).view(-1, 1, 1) 
        self.std = torch.FloatTensor(cnn_normalization_std).view(-1, 1, 1).to(self.device)
        #print(self.std)
 
    def forward(self, img):
        # normalize img
        return ((img - self.mean) / self.std).to(self.device)

################################################3
class quasy_conv_01_soft(Layer_01):
    def __init__( self , n_size=11, canal_in_0=3, cannal_out_0=5,last_act='relu',\
                 L1 = 0., L2 = 0.,device = None,show=0):
        super(quasy_conv_01_soft , self).__init__()         
        self.class_name = self.__class__.__name__        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.L1=L1
        self.L2=L2
        self.add_module('conv_layer_universal_uno_100', conv_layer_universal_uno_05\
                        (canal_in_0, n_size**2, 1,'relu',n_size,int(n_size/2),1, self.device)
                       )
         
        self.add_module('conv_layer_universal_uno_101', conv_layer_universal_uno_05\
                        (int(n_size**2 ), cannal_out_0, 0,last_act,1,0,1, self.device)
                       )

        
        
        self.to(self.device)
        self.reset_parameters()
        
    ###############################################################        
    def forward(self, im):
         
         
         
         
        im_100=self._modules['conv_layer_universal_uno_100'](im)         
         
        im_104=self._modules['conv_layer_universal_uno_101'](im_100)
         

        
        y = self._contiguous(im_104 )
        return y
################################################3
class vgg_19_decomposition_01():
    def __init__(self,device = None):
        self.device=device
        self.style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

        
        self.vgg_normalization = Normalization( self.device) 
        #self.vgg_conv = torchvision.models.vgg19(True).to(self.device)

        vgg_conv = torchvision.models.vgg19(pretrained=0)##конв часть+ векторизация
        vgg_conv.load_state_dict(torch.load('vgg19-dcbb9e9d.pth', map_location=self.device))

        self.vgg_conv= vgg_conv.features.to(self.device)# только  конв часть

        self.vgg_vect=vgg_conv.to(self.device)
        for param in self.vgg_conv.parameters():
            param.requires_grad = False
        for param in self.vgg_vect.parameters():
            param.requires_grad = False
            ###################################################################
    def load_image(self,img_path, max_size=512, shape=None):
        ''' Load in and transform an image, making sure the image
           is <= 400 pixels in the x-y dims.'''

        image = Image.open(img_path).convert('RGB')

        # large images will slow down processing
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if shape is not None:
            size = shape

        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                 (0.229, 0.224, 0.225))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)

        return image
    
    def get_features(self,image,   layers=None):
        """ Run an image forward through a model and get the features for 
            a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """
        model=self.vgg_conv
        ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
        ## Need the layers for the content and style representations of an image
        if layers is None:
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1', 
                      '10': 'conv3_1', 
                      '19': 'conv4_1',
                      '21': 'conv4_2',  ## content representation
                      '28': 'conv5_1'}


        ## -- do not need to change the code below this line -- ##
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            #print( name)
            #print(isinstance(x, (torch.Tensor)))
            
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features
    def gram_matrix(self,tensor):
        """ Calculate the Gram Matrix of a given tensor 
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """

        ## get the batch_size, depth, height, and width of the Tensor
        ## reshape it, so we're multiplying the features for each channel
        ## calculate the gram matrix
        a, d, h, w = tensor.size()
        #print('tensor.size()',tensor.size())
        # reshape so we're multiplying the features for each channel
        
        tensor = tensor.reshape([a*d, h * w])
        #tensor = tensor.view(a*d, h * w)
        # calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())
        return gram
    def style_loss_0(self, features,features1):
        style_grams = {layer: self.gram_matrix(features[layer]) for layer in features}
        style_grams1 = {layer: self.gram_matrix(features1[layer]) for layer in features1}
        style_loss = 0
        for layer in self.style_weights:
            style_gram = style_grams[layer]
            style_gram1 = style_grams1[layer]
            _, d, h, w = features[layer].shape
            #print(features[layer].shape,d * h * w)
            layer_style_loss = self.style_weights[layer] * torch.mean((style_gram - style_gram1)**2)
            #print(layer_style_loss/ (d * h * w))
            style_loss += layer_style_loss / (d * h * w)
        return style_loss
    def content_loss_0( self,features,features1):
        content_tenzor=features['conv4_2'] 
        content_tenzor1=features1['conv4_2']
        content_loss = torch.mean((content_tenzor - content_tenzor1)**2)
        return content_loss
    def total_loss_0(self, features,features1):
        # you may choose to leave these as is
        content_weight = 1  # alpha
        style_weight = 700#1e6  # bet

        content_loss=self.content_loss_0( features,features1)
        style_loss=self.style_loss_0( features,features1)
        total_loss = content_weight * content_loss + style_weight * style_loss
        #print('content_loss',content_loss)
        #print('style_loss',style_loss)
        return total_loss
    def tensor_total_loss_0(self,tenzor_1,tenzor_2):
        features=self.get_features(tenzor_1 ,  layers=None)
        features1=self.get_features(tenzor_2 ,   layers=None)
        return self.total_loss_0(features,features1)
        
######################################################################

 
##################################################3 
def map_00(txtr_,routine):
    a, b, c, d = txtr_.size()  # a=batch size(=1)        
    features =txtr_.reshape([a * b, c * d])
    G = torch.mm(features, features.t())  # compute the gram product


    if routine:
        (U, S, V)=torch.pca_lowrank(G, q=10, center=0, niter=2)
        map_2=torch.mm(U, U.t())
    else:
        map_2=G.div(a * b * c * d)
    return map_2


class mul_layer_0(Layer_01):
    def __init__(self, cannals_,cannals_txt_of_str , L1 = 0., L2 = 0.,device = None):
        super(mul_layer_0, self).__init__()
        
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.class_name = self.__class__.__name__        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.regularizer = Regularizer(L1, L2)
#        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
#        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
#        
#        self.add_module('permut_channelfirst', _layer_permut_channelfirst)
#        self.add_module('permut_channellast', _layer_permut_channellast)
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        self.cannals_txt_of_str=cannals_txt_of_str
        ###################################################
        self.add_module('conv_layer_universal_uno_060', conv_layer_universal_uno_05\
                (cannals_, int(cannals_/2), 1,'relu',3,1,1, self.device)
               )
        self.add_module('conv_layer_universal_uno_061', conv_layer_universal_uno_05\
                (cannals_, int(cannals_/2), 1,'relu',3,1,1, self.device)
               )
        self.add_module('conv_layer_universal_uno_062', conv_layer_universal_uno_05\
                (cannals_, int(cannals_/2), 1,'relu',3,1,1, self.device)
               )
        self.add_module('permt_canals_060', conv_layer_universal_uno_05\
                (3*int(cannals_/2), int(cannals_/2), 1,'relu',1,0,1, self.device)
               )

        self.add_module('conv_layer_universal_uno_501', conv_layer_universal_uno_05\
                (cannals_, cannals_txt_of_str, 1,'relu',3,1,1, self.device)
               )
        self.conv_layer_upsample_60=conv_layer_universal_upsample_05\
                 (int(cannals_/2)+cannals_txt_of_str, int(cannals_/2), 3,  True,  L1,  L2, self.device )  
        
        self.to(self.device)
        self.reset_parameters()
        
    def gram_matrix(self, input_):
        a, b, c, d = input_.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features =input_.reshape([a * b, c * d])
        #features = input_.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
        
    def marrige_00(self,struct2_1,texture_layer, name_reduse_cannal):
        #print('texture_layer',texture_layer.shape)
        if 0:
            progection_txtr=self.gram_matrix(texture_layer[:1,: ,: ])
        else:
            progection_txtr=map_00(texture_layer[:1,: ,: ],1)

        #print('progection_txtr',progection_txtr.shape)
        struct2_1_progected=self._layer_permut_channellast(struct2_1)
        struct2_1_progected=torch.reshape(struct2_1_progected, (-1, struct2_1_progected.shape[1],struct2_1_progected.shape[2],1,struct2_1_progected.shape[3]))
        struct2_1_progected=torch.squeeze(torch.matmul(struct2_1_progected,progection_txtr),-2)
        struct2_1_progected=self._layer_permut_channelfirst(struct2_1_progected)        
        struct2_1_mapped=self._modules[name_reduse_cannal](struct2_1_progected) 
        return  struct2_1_mapped
         
    def forward(self,struct3_1,struct3_1_txtr ):
        [a1,a2,a3,a4] = struct3_1_txtr.shape
        textre_10=self.marrige_00( struct3_1,struct3_1_txtr[:,:,::int(a3/5),::int(a4/5)], 'conv_layer_universal_uno_060')
        textre_11=self.marrige_00( struct3_1,struct3_1_txtr[:,:,::int(a3/10),::int(a4/10)], 'conv_layer_universal_uno_061')
        textre_12=self.marrige_00( struct3_1,struct3_1_txtr[:,:,::int(a3/20),::int(a4/20)], 'conv_layer_universal_uno_062')
        struct3_1_progected_txtr=torch.cat((textre_10, textre_11 ,textre_12   ),axis=1)
        struct3_1_progected_txtr=self._modules['permt_canals_060'](struct3_1_progected_txtr)
        
        betta_0=self._modules['conv_layer_universal_uno_501'](struct3_1)
        struct3_1_progected_txtr=torch.cat((betta_0,struct3_1_progected_txtr  ),axis=1)
        betta_1=self._modules['conv_layer_upsample_60'](struct3_1_progected_txtr)#torch.Size([2, 128, 128, 256])

        

        return betta_1
    def _get_regularizer(self):
        return self.regularizer
####################################################################################################################
class Dataset_12_TP_train(Dataset):
    def __init__(self, base0 ,shape_=[256,512], routine_deform=1, cannals=1):
        self.base0 = base0 
        self.length=len(self.base0.df)
        train_part=0.8
        test_part=0.2
        self.length_train = int(self.length * train_part)
        self.length_test = int(self.length * test_part)
        self.shape_=shape_ 
        self.cannals=cannals
        xv, yv = np.meshgrid(range(shape_[0]), range(shape_[1]), sparse=False, indexing='ij')
        xv=2*(xv/shape_[0]-0.5)
        yv=2*(yv /shape_[1]-0.5)
        if routine_deform:
            poly_2d_x=  30*np.sin(apply_poly_00(xv,[0,-4,50,10,5])/15)    
            poly_2d_y=  30*np.cos(apply_poly_00(yv,[0,-27,-28,-39])/15 ) 
        else:# нет деформации
            poly_2d_x=   apply_poly_00(xv,[0,0,0,0,0]) 
            poly_2d_y=   apply_poly_00(yv,[0,0,0,0]) 

        self.routine_deform=routine_deform
        self.mot_xy=1*np.concatenate([np.expand_dims(poly_2d_x,2),np.expand_dims(poly_2d_y,2)],2) 
        if 0:
            show_surf_00(xv, yv, poly_2d_x+poly_2d_y)
        self.multiskotch_=multiskotch_01(bit_per_pixel=2,rotine_sketch=0) 

        
    def set_train_mode(self):
        self.test_mode = False
        self.length = self.length_train
 
    def set_test_mode(self):
        self.test_mode = True
        self.length = self.length_test
 
    def __len__(self):
        return len(self.base0.df)
    ##########################################################################3
    def triplet_TL_00(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 0:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image, (self.shape_[1],self.shape_[0]))
        image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        self.multiskotch_.sketch_multi(image_YUV)
        y_sketch=self.multiskotch_.multi_sketch 
 

        return image_YUV/255,  y_sketch
    def triplet_TL_01(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 1:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image_RGB, (self.shape_[1],self.shape_[0]))
         
        self.multiskotch_.sketch_multi(image)
        y_sketch=self.multiskotch_.multi_sketch 
 

        return image/255,  y_sketch
 
    def __getitem__(self, idx):
        path=self.base0.path_in
        image_file_cur= self.base0.df.iloc[idx]['cur'] 
        image_file_ref= self.base0.df.iloc[idx]['ref'] 
         
        f_name_cur= os.path.join(path, image_file_cur) 
        f_name_ref= os.path.join(path, image_file_ref) 
        #print(f_name)
        im_gadol, sketch=self.triplet_TL_01(f_name_cur) 
        im_gadol_ref, sketch_ref=self.triplet_TL_01(f_name_ref) 
        if self.routine_deform:
            sketch_1= shift_flow_T_01(sketch_ref ,self.mot_xy)
            im_gadol_1=shift_flow_T_01(im_gadol_ref ,self.mot_xy)
        else:
            sketch_1= sketch_ref
            im_gadol_1=im_gadol_ref
            

        sketch01 = sketch  
        if self.cannals==3:
            sketch_cur=np.concatenate([sketch01,sketch01,sketch01],-1)
                  
        if self.cannals==2:
            canal=np.mod(idx,3)
            sketch_cur=sketch01             
            im_gadol=np.expand_dims(im_gadol[:,:,canal],2)
            im_gadol_1=np.expand_dims( im_gadol_1[:,:,canal],2)

        elif self.cannals==1:
            sketch_cur=sketch01
            im_gadol=np.expand_dims(np.mean(im_gadol,-1),2)
            im_gadol_1=np.expand_dims(np.mean(im_gadol_1,-1),2)

         
         
        return {
            'images_cur': im_gadol,             
            'sketch_cur':sketch_cur,
            'images_ref':im_gadol_1,
            'skotch_shift':sketch_1
            
        }
################################################################
####################################################################################################################
class Dataset_12_TP_test(Dataset):
    def __init__(self, video_in ,shape_=[256,512],routine_deform=0, cannals=1):
        files_ = sorted(os.listdir(video_in ))                 
        l_=len(files_)
        #print('len',l_)
        if l_>0:
            file_=files_[0]
            if file_.endswith(".jpg") or file_.endswith(".png") or file_.endswith(".jpeg"):
                files_cur = [video_in+files_[i] for i in range(l_)]
        self.image_file_cur=files_cur
        self.image_file_ref=video_in+file_
        
        
        self.length=l_
        train_part=0.8
        test_part=0.2
        self.length_train = int(self.length * train_part)
        self.length_test = int(self.length * test_part)
        self.shape_=shape_ 
        self.cannals=cannals
        xv, yv = np.meshgrid(range(shape_[0]), range(shape_[1]), sparse=False, indexing='ij')
        xv=2*(xv/shape_[0]-0.5)
        yv=2*(yv /shape_[1]-0.5)
        if routine_deform:
            poly_2d_x=  30*np.sin(apply_poly_00(xv,[0,-4,50,10,5])/15)    
            poly_2d_y=  30*np.cos(apply_poly_00(yv,[0,-27,-28,-39])/15 ) 
        else:# нет деформации
            poly_2d_x=   apply_poly_00(xv,[0,0,0,0,0]) 
            poly_2d_y=   apply_poly_00(yv,[0,0,0,0]) 


        self.mot_xy=1*np.concatenate([np.expand_dims(poly_2d_x,2),np.expand_dims(poly_2d_y,2)],2) 
        if 0:
            show_surf_00(xv, yv, poly_2d_x+poly_2d_y)
        self.multiskotch_=multiskotch_01(bit_per_pixel=2,rotine_sketch=0) 

        
    def set_train_mode(self):
        self.test_mode = False
        self.length = self.length_train
 
    def set_test_mode(self):
        self.test_mode = True
        self.length = self.length_test
 
    def __len__(self):
        return self.length
    ##########################################################################3
    def triplet_TL_00(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 0:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image, (self.shape_[1],self.shape_[0]))
        image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        self.multiskotch_.sketch_multi(image_YUV)
        y_sketch=self.multiskotch_.multi_sketch 
 

        return image_YUV/255,  y_sketch
    def triplet_TL_01(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 1:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image_RGB, (self.shape_[1],self.shape_[0]))
         
        self.multiskotch_.sketch_multi(image)
        y_sketch=self.multiskotch_.multi_sketch 
 

        return image/255,  y_sketch
 
    def __getitem__(self, idx):
         
        f_name_cur= self.image_file_cur[idx] 
        f_name_ref= self.image_file_ref
        if 0:
            print(idx)
            print(f_name_cur)
            print(f_name_ref)
            print('-----------------')
         



        
        im_gadol, sketch=self.triplet_TL_01(f_name_cur) 
        im_gadol_ref, sketch_ref=self.triplet_TL_01(f_name_ref) 
        if 0:
            sketch_1= shift_flow_T_01(sketch_ref ,self.mot_xy)
            im_gadol_1=shift_flow_T_01(im_gadol_ref ,self.mot_xy)
        else:
            sketch_1= sketch_ref
            im_gadol_1=im_gadol_ref
            

        sketch01 = sketch  
        if self.cannals==3:
            sketch_cur=np.concatenate([sketch01,sketch01,sketch01],-1)
                  
        if self.cannals==4:
            canal=0
            sketch_cur=sketch01             
            im_gadol=np.expand_dims(im_gadol[:,:,canal],2)
            im_gadol_1=np.expand_dims( im_gadol_1[:,:,canal],2)
        if self.cannals==5:
            canal=1
            sketch_cur=sketch01             
            im_gadol=np.expand_dims(im_gadol[:,:,canal],2)
            im_gadol_1=np.expand_dims( im_gadol_1[:,:,canal],2)
        if self.cannals==6:
            canal=3
            sketch_cur=sketch01             
            im_gadol=np.expand_dims(im_gadol[:,:,canal],2)
            im_gadol_1=np.expand_dims( im_gadol_1[:,:,canal],2)
            
        if self.cannals==2:
            canal=np.mod(idx,3)
            sketch_cur=sketch01             
            im_gadol=np.expand_dims(im_gadol[:,:,canal],2)
            im_gadol_1=np.expand_dims( im_gadol_1[:,:,canal],2)

        elif self.cannals==1:
            sketch_cur=sketch01
            im_gadol=np.expand_dims(np.mean(im_gadol,-1),2)
            im_gadol_1=np.expand_dims(np.mean(im_gadol_1,-1),2)

         
         
        return {
            'images_cur': im_gadol,             
            'sketch_cur':sketch_cur,
            'images_ref':im_gadol_1,
            'skotch_shift':sketch_1
            
        }
##########################################################

#####################################################################3
class atrous_pyramid_00(Layer_01):
    def __init__(self,num_filteres_in=3, num_filteres_out=1,num_filteres_middle=1, L1 = 0., L2 = 0.,device = None,show=0):
        super( atrous_pyramid_00 , self).__init__()         
        self.class_name = self.__class__.__name__        
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L1=L1
        self.L2=L2
        self.show=show
        self.regularizer = Regularizer(L1, L2)
        self.add_module('conv_layer_universal_uno_05_00',conv_layer_universal_uno_05( 
                 num_filteres_in, num_filteres_middle, 1,'relu',5,2,1,self.device  
                    ))
        self.add_module('conv_layer_universal_uno_05_01',conv_layer_universal_uno_05( 
                 num_filteres_in, num_filteres_middle, 1,'relu',5,4,2,self.device                                                               
                                                                                 ))
        self.add_module('conv_layer_universal_uno_05_02',conv_layer_universal_uno_05( 
                 num_filteres_in, num_filteres_middle, 1,'relu',5,6,3,self.device                                                               
                                                                                 ))
        self.add_module('conv_layer_universal_uno_05_03',conv_layer_universal_uno_05( 
                 num_filteres_in, num_filteres_middle, 1,'relu',5,8,4,self.device 
            ))
        p_z=0 
        k_z=1
        bias_=1
        _layer_conv_31 = Conv2d(4*num_filteres_middle, 8*num_filteres_middle, kernel_size=(k_z, k_z),
                                stride=(1, 1), padding = (p_z, p_z), padding_mode = 'zeros', bias=bias_)
        self.add_module('conv_31', _layer_conv_31)     
        _layer_activation_31 = LeakyReLU(0.05)  
        self.add_module('activation_31', _layer_activation_31)
        _layer_conv_41 = Conv2d(8*num_filteres_middle, num_filteres_out, kernel_size=(k_z, k_z),
                                stride=(1, 1), padding = (p_z, p_z), padding_mode = 'zeros', bias=bias_)
        
        self.add_module('conv_41', _layer_conv_41) 
        _layer_batch_norm_3 = BatchNorm2d(num_filteres_out)
        self.add_module('batch_norm_3', _layer_batch_norm_3)
        _layer_activation_41 = LeakyReLU(0.05)  
        self.add_module('activation_41', _layer_activation_41)

        
        
        self.to(self.device)
        self.reset_parameters()
        
    ###############################################################        
    def forward(self, ref0):
         
         
         
         
        q00= self._modules['conv_layer_universal_uno_05_00'](ref0)
        q01= self._modules['conv_layer_universal_uno_05_01'](ref0)
        q02= self._modules['conv_layer_universal_uno_05_02'](ref0)
        q03= self._modules['conv_layer_universal_uno_05_03'](ref0)
        merged_00 = torch.cat((q00, q01,q02, q03 ),axis=1)
        if self.show:
            print('merged_00',merged_00.shape)
        q04= self._modules['conv_31'](merged_00)
        q05= self._modules['activation_31'](q04)          
        q06= self._modules['conv_41'](q05) 
        q07= self._modules['batch_norm_3'](q06)          
        q08= self._modules['activation_41'](q07) 
        if self.show:
            print('q05 ',q05.shape)
            print('q08 ',q08.shape)
        
        y = self._contiguous(q08)
        return y
################################################3

################################################################3
def kernel_Ntxtr(k):
    a=np.eye((k**2))
    b=np.reshape(a,[-1,1,k,k])
    return b
def int_log_2(x):
    return  int(np.log(x ) /np.log(2 ) )
class im2txtrTnzr02(Layer_09):
    def __init__(self, param, device=None):
         
        super( im2txtrTnzr02, self).__init__(  param["imageSize"]) 
        modules = []
        
        self.param = param
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape = tuple(param["imageSize"])
        self.size_layers = tuple(param["w_"])
        for w_ in self.size_layers:
            W5 =  kernel_Ntxtr(w_)
            step_=max(1,int(w_**2/500))
            W5=W5[::step_,...]
            out_channels_=W5.shape[0]
            strides_=2**int_log_2(w_)
            layer_conv_64 = torch.nn.Conv2d(in_channels=self.input_shape[-1], out_channels=w_**2, kernel_size=(w_,w_),\
                                        stride =(strides_,strides_), padding = (int(w_/2)-1, int(w_/2)-1), padding_mode = 'zeros', bias=0)
            layer_conv_64.weight.data = torch.FloatTensor(W5)             
            self.add_module('conv_'+str(w_), layer_conv_64) 
 
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))         
        self.to(self.device)
        
    def get_features(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
        x = self._layer_permut_channelfirst(x)
        features = {}
        for w_ in self.size_layers:
                    
            x_=self._modules['conv_'+str(w_)] (x)
            features[str(w_)]=x_
        
        
         
 
        return features



    def get_features_01(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
       
        features = {}
        for w_ in self.size_layers:
                    
            x_=self._modules['conv_'+str(w_)] (x)
            features[str(w_)]=x_
        
        
         
 
        return features

    def map_00(self,txtr_,routine):
        a, b, c, d = txtr_.size()  # a=batch size(=1)        
        features =txtr_.reshape([a * b, c * d])
        G = torch.mm(features, features.t())  # compute the gram product


        if routine:
            (U, S, V)=torch.pca_lowrank(G, q=10, center=0, niter=2)
            map_2=torch.mm(U, U.t())
        else:
            map_2=G.div(a * b * c * d)
        return map_2
 
    def map_24(self,x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
        x = self._layer_permut_channelfirst(x)
        map_ = {}
        for w_ in self.size_layers:
             
            x24=self._modules['conv_'+str(w_)] (x)
            map_0=self.map_00(x24,1)
            map_[str(w_)]=map_0
         
         
        
         
        return map_
    
    
    
    def get_features_64(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
        x = self._layer_permut_channelfirst(x)
         
        x64=self.conv_64(x)         
        
        
        
        
        
        return x64
        
    def forward(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        
        x = self.conv_64(x)
        print(x.shape)
        if (to_numpy):
            if (self.device  == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()

        if (to_list):
            if (self.device  == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        return x
    
    def summary(self):
        _summary(self, input_size = self.input_shape, device = self.device)
        
    def ONNXexport(self, filename, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            x = torch.FloatTensor(x).to(self.device)
        torch.onnx.export(self, x, filename, export_params=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
 

class fully_connect_feller_10(Layer_09):
    def __init__(self, Size_, last_activate, L1 = 0., L2 = 0.,device = None ):
        super(fully_connect_feller_10, self).__init__([Size_[0]] )    

        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.class_name = self.__class__.__name__
        self.last_activate = last_activate

        self.Size = Size_ 
        self.regularizer = Regularizer(L1, L2)
        self.L1=L1
        self.L2=L2
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ####################################### 
        w_prewious=self.Size[0]
        for w_ in self.Size[1:] :
            _layer_D00 = Linear( w_prewious,  w_, bias = True)
            self.add_module('D_'+str(w_), _layer_D00) 
            w_prewious=w_
            
        for w_ in self.Size[1:3] :
            _layer_Dropout00 = Dropout(0.1)
            self.add_module('Dropout00_'+str(w_), _layer_Dropout00) 
            #print('drop','Dropout00_'+str(w_))
        _layer_activation_LW1 = LeakyReLU(0.05)             
        self.add_module('activation_LW1' ,  _layer_activation_LW1)          
        if last_activate == 'sigmoid':
            _layer_activation_D4 = Sigmoid()
            self.add_module('activation_D4', _layer_activation_D4)
        self.to(self.device)
    #####################################################
    def forward(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        
        x = self._contiguous(x)
        count=0
        for w_ in self.Size[1:] :
            x=self._modules['D_'+str(w_)](x)
            if  count<2:
                x=self._modules['Dropout00_'+str(w_)](x)
            x=self._modules['activation_LW1'](x) 
            count+=1
        if self.last_activate == 'sigmoid':
            x= self._modules['activation_D4'](x) 
        
        

        return x

         
        
         
    def _get_regularizer(self):
        return self.regularizer

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
class mul_layer_5(Layer_01):
    def __init__(self, cannals_,  cannals_txt_of_str,L1 = 0., L2 = 0.,device = None):
        super(mul_layer_5, self).__init__()
        
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.cannals_txt_of_str=cannals_txt_of_str
        self.class_name = self.__class__.__name__        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

#        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
#        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
#        
#        self.add_module('permut_channelfirst', _layer_permut_channelfirst)
#        self.add_module('permut_channellast', _layer_permut_channellast)
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
      
        self.regularizer = Regularizer(L1, L2)
        ###################################################
        self.add_module('conv_layer_universal_uno_060', conv_layer_universal_uno_05\
                (cannals_, int(cannals_/2), 1,'relu',3,1,1, self.device)
               )
        self.add_module('conv_layer_universal_uno_061', conv_layer_universal_uno_05\
                (cannals_, int(cannals_/2), 1,'relu',3,1,1, self.device)
               )
         
        self.add_module('permt_canals_060', conv_layer_universal_uno_05\
                (cannals_, cannals_, 1,'relu',1,0,1, self.device)
               )
        if  self.cannals_txt_of_str>0:
            self.add_module('conv_layer_universal_uno_501', conv_layer_universal_uno_05\
                    (cannals_, cannals_txt_of_str, 1,'relu',3,1,1, self.device)
                   )

        self.conv_layer_uno_60=conv_layer_universal_uno_05\
                 (cannals_+cannals_txt_of_str, cannals_, 1,'relu',3,1,1, self.device)  
        
        
        self.to(self.device)
        self.reset_parameters()
    def map_00(self, txtr_,routine):
        a, b, c, d = txtr_.size()  # a=batch size(=1)        
        features =txtr_.reshape([a * b, c * d])
        G = torch.mm(features, features.t())  # compute the gram product


        if routine:
            (U, S, V)=torch.pca_lowrank(G, q=10, center=0, niter=2)
            map_2=torch.mm(U, U.t())
        else:
            map_2=G.div(a * b * c * d)
        return map_2

    def gram_matrix(self, input_):
        a, b, c, d = input_.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features =input_.reshape([a * b, c * d])
        #features = input_.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
        
    def marrige_00(self,struct2_1,texture_layer, name_reduse_cannal):
        #print('texture_layer',texture_layer.shape)
        #progection_txtr=self.gram_matrix(texture_layer[:1,: ,: ])
        if 0:
            progection_txtr=self.gram_matrix(texture_layer[:1,: ,: ])
        else:
            progection_txtr=self.map_00(texture_layer[:1,: ,: ],1)

         
        #print('progection_txtr',progection_txtr.shape)
        struct2_1_progected=self._layer_permut_channellast(struct2_1)
        struct2_1_progected=torch.reshape(struct2_1_progected, (-1, struct2_1_progected.shape[1],struct2_1_progected.shape[2],1,struct2_1_progected.shape[3]))
        struct2_1_progected=torch.squeeze(torch.matmul(struct2_1_progected,progection_txtr),-2)
        struct2_1_progected=self._layer_permut_channelfirst(struct2_1_progected)        
        struct2_1_mapped=self._modules[name_reduse_cannal](struct2_1_progected) 
        return  struct2_1_mapped
         
    def forward(self,struct3_1,struct3_1_txtr ):
        [a1,a2,a3,a4] = struct3_1_txtr.shape
        textre_10=self.marrige_00( struct3_1,struct3_1_txtr[:,:,::int(a3),::int(a4)], 'conv_layer_universal_uno_060')
        textre_11=self.marrige_00( struct3_1,struct3_1_txtr[:,:,::int(a3/2),::int(a4/2)], 'conv_layer_universal_uno_061')
        
        struct3_1_progected_txtr=torch.cat((textre_10, textre_11   ),axis=1)
        struct3_1_progected_txtr=self._modules['permt_canals_060'](struct3_1_progected_txtr)#torch.Size([2, 32, 256, 512]
         
        if  self.cannals_txt_of_str>0:
            betta_0=self._modules['conv_layer_universal_uno_501'](struct3_1)#torch.Size([2, 12, 256, 512])

            struct3_1_progected_txtr=torch.cat((betta_0,struct3_1_progected_txtr  ),axis=1)
         
         
         
        betta_1=self._modules['conv_layer_uno_60'](struct3_1_progected_txtr)#torch.Size([2, 44, 256, 512])
         
        
        return betta_1
    def _get_regularizer(self):
        return self.regularizer
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
class mul_layer_6(Layer_01):
    def __init__(self, cannals_,  cannals_txt_of_str,L1 = 0., L2 = 0.,device = None):
        super(mul_layer_6, self).__init__()
        
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.cannals_txt_of_str=cannals_txt_of_str
        self.class_name = self.__class__.__name__        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

#        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
#        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
#        
#        self.add_module('permut_channelfirst', _layer_permut_channelfirst)
#        self.add_module('permut_channellast', _layer_permut_channellast)
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
      
        self.regularizer = Regularizer(L1, L2)
        ###################################################
        self.add_module('conv_layer_universal_uno_060', conv_layer_universal_uno_05\
                (cannals_, int(cannals_/2), 1,'relu',3,1,1, self.device)
               )
        self.add_module('conv_layer_universal_uno_061', conv_layer_universal_uno_05\
                (cannals_, int(cannals_/2), 1,'relu',3,1,1, self.device)
               )
        self.add_module('conv_layer_universal_uno_062', conv_layer_universal_uno_05\
                (cannals_, int(cannals_/2), 1,'relu',3,1,1, self.device)
               )
         
        self.add_module('permt_canals_060', conv_layer_universal_uno_05\
                (int(3*cannals_/2), cannals_, 1,'relu',1,0,1, self.device)
               )
        if  self.cannals_txt_of_str>0:
            self.add_module('conv_layer_universal_uno_501', conv_layer_universal_uno_05\
                    (cannals_, cannals_txt_of_str, 1,'relu',3,1,1, self.device)
                   )

        self.conv_layer_uno_60=conv_layer_universal_uno_05\
                 (cannals_+cannals_txt_of_str, cannals_, 1,'relu',3,1,1, self.device)  
        
        
        self.to(self.device)
        self.reset_parameters()
    def map_00(self, txtr_,routine):
        a, b, c, d = txtr_.size()  # a=batch size(=1)        
        features =txtr_.reshape([a * b, c * d])
        G = torch.mm(features, features.t())  # compute the gram product


        if routine:
            (U, S, V)=torch.pca_lowrank(G, q=10, center=0, niter=2)
            map_2=torch.mm(U, U.t())
        else:
            map_2=G.div(a * b * c * d)
        return map_2

    def gram_matrix(self, input_):
        a, b, c, d = input_.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features =input_.reshape([a * b, c * d])
        #features = input_.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
        
    def marrige_00(self,struct2_1,texture_layer, name_reduse_cannal):
        #print('texture_layer',texture_layer.shape)
        #progection_txtr=self.gram_matrix(texture_layer[:1,: ,: ])
        if 0:
            progection_txtr=self.gram_matrix(texture_layer[:1,: ,: ])
        else:
            progection_txtr=self.map_00(texture_layer[:1,: ,: ],1)

         
        #print('progection_txtr',progection_txtr.shape)
        struct2_1_progected=self._layer_permut_channellast(struct2_1)
        struct2_1_progected=torch.reshape(struct2_1_progected, (-1, struct2_1_progected.shape[1],struct2_1_progected.shape[2],1,struct2_1_progected.shape[3]))
        struct2_1_progected=torch.squeeze(torch.matmul(struct2_1_progected,progection_txtr),-2)
        struct2_1_progected=self._layer_permut_channelfirst(struct2_1_progected)        
        struct2_1_mapped=self._modules[name_reduse_cannal](struct2_1_progected) 
        return  struct2_1_mapped
         
    def forward(self,struct3_1,struct3_1_txtr ):
        [a1,a2,a3,a4] = struct3_1_txtr.shape
        textre_10=self.marrige_00( struct3_1,struct3_1_txtr[:,:,::int(a3/3),::int(a4/3)], 'conv_layer_universal_uno_060')
        textre_11=self.marrige_00( struct3_1,struct3_1_txtr[:,:,::int(a3/7),::int(a4/7)], 'conv_layer_universal_uno_061')
        textre_12=self.marrige_00( struct3_1,struct3_1_txtr[:,:,::int(a3/15),::int(a4/15)], 'conv_layer_universal_uno_062')
        struct3_1_progected_txtr=torch.cat((textre_10, textre_11 ,textre_12  ),axis=1)
        struct3_1_progected_txtr=self._modules['permt_canals_060'](struct3_1_progected_txtr)#torch.Size([2, 32, 256, 512]
         
        if  self.cannals_txt_of_str>0:
            betta_0=self._modules['conv_layer_universal_uno_501'](struct3_1)#torch.Size([2, 12, 256, 512])

            struct3_1_progected_txtr=torch.cat((betta_0,struct3_1_progected_txtr  ),axis=1)
         
         
         
        betta_1=self._modules['conv_layer_uno_60'](struct3_1_progected_txtr)#torch.Size([2, 44, 256, 512])
         
        
        return betta_1
    def _get_regularizer(self):
        return self.regularizer

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
class mul_SA(Layer_01):
    def __init__(self, size0,size1, L1 = 0., L2 = 0.,device = None):
        super(mul_SA, self).__init__()
        
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.L1=L1
        self.L2=L2
         
        self.class_name = self.__class__.__name__        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

#        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
#        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
#        
#        self.add_module('permut_channelfirst', _layer_permut_channelfirst)
#        self.add_module('permut_channellast', _layer_permut_channellast)
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        self._layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
      
         
        ###################################################
        self.fully_connect_feller_0=    fully_connect_feller_10(size0, 'lin',self.L1, self.L2, self.device)
        self.fully_connect_feller_1=    fully_connect_feller_10(size1, 'lin',self.L1, self.L2, self.device)

        self.to(self.device)
        self.reset_parameters()
        
    def forward(self,im_512,scatch_64 ):
        im_512_lin_0=self._layer_permut_channellast(im_512)
        a0, b0, c0, d0 = im_512_lin_0.size()
        im_512_lin_1 =im_512_lin_0.reshape([a0,-1, d0])
        im_512_lin_2=self.fully_connect_feller_0(im_512_lin_1)#torch.Size([256, 32])
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        scatch_512_lin_0=self._layer_permut_channellast(scatch_64)
        a1, b1, c1, d1 = scatch_512_lin_0.size()
        scatch_512_lin_1 =scatch_512_lin_0.reshape([a1,-1, d1])
        scatch_512_lin_2=self.fully_connect_feller_1(scatch_512_lin_1)#torch.Size([256, 32])
        ########################################
        scatch_512_lin_3=self._layer_permut_channel_SA(scatch_512_lin_2)
        m_SA= torch.matmul(  im_512_lin_2,scatch_512_lin_3) 
        _layer_SfTMax = Softmax(dim = 2)
        m_SA=_layer_SfTMax(m_SA)
        deep_morphing_0=torch.matmul( m_SA,scatch_512_lin_1) 
        deep_morphing_1= deep_morphing_0.reshape([a0, b0, c0,d1])
        deep_morphing_2=self._layer_permut_channelfirst(deep_morphing_1)
        if 0:
            print('=====================')
            print(im_512_lin_0.shape)
            print(im_512_lin_1.shape)
            print(im_512_lin_2.shape)
            print(scatch_512_lin_0.shape)
            print(scatch_512_lin_1.shape)
            print(scatch_512_lin_2.shape)
            print(scatch_512_lin_3.shape)
            print(m_SA.shape)
            print(deep_morphing_1.shape)
            print('=====================')
         
        
        return deep_morphing_2
    def _get_regularizer(self):
        return self.regularizer
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
class mul_SA_01(Layer_01):
    def __init__(self, size0,size1, L1 = 0., L2 = 0.,device = None):
        super(mul_SA_01, self).__init__()
        
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.L1=L1
        self.L2=L2
         
        self.class_name = self.__class__.__name__        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

#        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
#        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
#        
#        self.add_module('permut_channelfirst', _layer_permut_channelfirst)
#        self.add_module('permut_channellast', _layer_permut_channellast)
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        self._layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
      
         
        ###################################################
        self.fully_connect_feller_0=    fully_connect_feller_10(size0, 'lin',self.L1, self.L2, self.device)
        self.fully_connect_feller_1=    fully_connect_feller_10(size1, 'lin',self.L1, self.L2, self.device)

        self.to(self.device)
        self.reset_parameters()
        
    def forward(self,im_512,scatch_64 ):
        im_512_lin_0=self._layer_permut_channellast(im_512)
        a0, b0, c0, d0 = im_512_lin_0.size()
        im_512_lin_1 =im_512_lin_0.reshape([a0,-1, d0])
        im_512_lin_2=self.fully_connect_feller_0(im_512_lin_1)#torch.Size([256, 32])
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        scatch_512_lin_0=self._layer_permut_channellast(scatch_64)
        a1, b1, c1, d1 = scatch_512_lin_0.size()
        scatch_512_lin_1 =scatch_512_lin_0.reshape([a1,-1, d1])
        scatch_512_lin_2=self.fully_connect_feller_1(scatch_512_lin_1)#torch.Size([256, 32])
        ########################################
        scatch_512_lin_3=self._layer_permut_channel_SA(scatch_512_lin_2)
        m_SA= torch.matmul(  im_512_lin_2,scatch_512_lin_3) 
        _layer_SfTMax = Softmax(dim = 2)
        m_SA=_layer_SfTMax(m_SA)
        deep_morphing_0=torch.matmul( m_SA,scatch_512_lin_1) 
        deep_morphing_1= deep_morphing_0.reshape([a0, b0, c0,d1])
        deep_morphing_2=self._layer_permut_channelfirst(deep_morphing_1)
        if 0:
            print('=====================')
            print(im_512_lin_0.shape)
            print(im_512_lin_1.shape)
            print(im_512_lin_2.shape)
            print(scatch_512_lin_0.shape)
            print(scatch_512_lin_1.shape)
            print(scatch_512_lin_2.shape)
            print(scatch_512_lin_3.shape)
            print(m_SA.shape)
            print(deep_morphing_1.shape)
            print('=====================')
         
        
        return deep_morphing_2,m_SA
    def _get_regularizer(self):
        return self.regularizer
####################################################################################################
class im2txtrTnzr03(Layer_09):
    def __init__(self, param, device=None):
         
        super( im2txtrTnzr03, self).__init__(  param["imageSize"]) 
        modules = []
        
        self.param = param
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
        self.input_shape = tuple(param["imageSize"])
        self.dilation = tuple(param["dilation"])
        self.size_layers = tuple(param["w_"])
        for w_ , dl_z in zip(self.size_layers,self.dilation):
            W5 =  kernel_Ntxtr(w_)
            step_=max(1,int(w_**2/500))
            W5=W5[::step_,...]
            out_channels_=W5.shape[0]
            strides_=2**int_log_2(w_)
            layer_conv_64 = torch.nn.Conv2d(in_channels=self.input_shape[-1], out_channels=w_**2, kernel_size=(w_,w_),\
                                        stride =(strides_,strides_),\
                                            dilation = (dl_z,dl_z), \
                                            padding = (dl_z*int(w_/2) -dl_z , dl_z*int(w_/2)-dl_z ),\
                                            padding_mode = 'zeros', bias=0)
             
            layer_conv_64.weight.data = torch.FloatTensor(W5)             
            self.add_module('conv_'+str(w_)+'_'+str(dl_z), layer_conv_64) 
 
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))         
        self.to(self.device)
        
    def get_features(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
        x = self._layer_permut_channelfirst(x)
        features = {}
        for w_ , dl_z in zip(self.size_layers,self.dilation):
                    
            x_=self._modules['conv_'+str(w_)+'_'+str(dl_z)] (x)
            features[str(w_)+'_'+str(dl_z)]=x_
        
        
         
 
        return features
    def represent_features(self,features ):
         
        for w_ , dl_z in zip(self.size_layers,self.dilation):
                    
             
            print( features[str(w_)+'_'+str(dl_z)].shape )
            q=np.mean(features[str(w_)+'_'+str(dl_z)].detach().numpy(),1)[0,...]
             
            
            ai_2(q)
        
        
         
 
        return 1



    def get_features_01(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
       
        features = {}
        for w_ , dl_z in zip(self.size_layers,self.dilation):
                    
            x_=self._modules['conv_'+str(w_)+'_'+str(dl_z)] (x)
            features[str(w_)+'_'+str(dl_z)]=x_
        
        
         
 
        return features

    def map_00(self,txtr_,routine):
        a, b, c, d = txtr_.size()  # a=batch size(=1)        
        features =txtr_.reshape([a * b, c * d])
        G = torch.mm(features, features.t())  # compute the gram product


        if routine:
            (U, S, V)=torch.pca_lowrank(G, q=10, center=0, niter=2)
            map_2=torch.mm(U, U.t())
        else:
            map_2=G.div(a * b * c * d)
        return map_2
 
    def map_24(self,x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (torch.Tensor)):
            x = x.to(self.device)
        x = self._contiguous(x)
        x = self._layer_permut_channelfirst(x)
        map_ = {}
        for w_ , dl_z in zip(self.size_layers,self.dilation):
             
            x24=self._modules['conv_'+str(w_)+'_'+str(dl_z)] (x)
            map_0=self.map_00(x24,1)
            map_[str(w_)+'_'+str(dl_z)]=map_0
         
         
        
         
        return map_
    
    
    def forward(self, x):
        to_numpy = False
        to_list = False
        if isinstance(x, (np.ndarray)):
            to_numpy = True
            x = torch.FloatTensor(x).to(self.device)
        if isinstance(x, (list, tuple)):
            to_list = True
            x = torch.FloatTensor(x).to(self.device)
        
        x = self.conv_64_1(x)
        print(x.shape)
        if (to_numpy):
            if (self.device  == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()

        if (to_list):
            if (self.device  == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        return x
    
    def summary(self):
        _summary(self, input_size = self.input_shape, device = self.device)
        
    def ONNXexport(self, filename, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            x = torch.FloatTensor(x).to(self.device)
        torch.onnx.export(self, x, filename, export_params=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
#########################################################

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
class mul_SA_02(Layer_01):
    def __init__(self, size0,size1, L1 = 0., L2 = 0.,device = None):
        super(mul_SA_02, self).__init__()
        
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.L1=L1
        self.L2=L2
         
        self.class_name = self.__class__.__name__        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

#        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
#        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
#        
#        self.add_module('permut_channelfirst', _layer_permut_channelfirst)
#        self.add_module('permut_channellast', _layer_permut_channellast)
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        self._layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
        self._layer_SfTMax = Softmax(dim = 2)
         
        ###################################################
        self.fully_connect_feller_0=    fully_connect_feller_10(size0, 'lin',self.L1, self.L2, self.device)
        self.fully_connect_feller_1=    fully_connect_feller_10(size1, 'lin',self.L1, self.L2, self.device)

        self.to(self.device)
        self.reset_parameters()
        
    def forward(self,im_512,scatch_64, im_refer_512 ):
        im_512_lin_0=self._layer_permut_channellast(im_512)
        a0, b0, c0, d0 = im_512_lin_0.size()
        im_512_lin_1 =im_512_lin_0.reshape([a0,-1, d0])
        im_512_lin_2=self.fully_connect_feller_0(im_512_lin_1)#torch.Size([256, 32])
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        scatch_512_lin_0=self._layer_permut_channellast(scatch_64)
        a1, b1, c1, d1 = scatch_512_lin_0.size()
        scatch_512_lin_1 =scatch_512_lin_0.reshape([a1,-1, d1])
        scatch_512_lin_2=self.fully_connect_feller_1(scatch_512_lin_1)#torch.Size([256, 32])
        ########################################
        im_refer_512_lin_0=self._layer_permut_channellast(im_refer_512)
        a2, b2, c2, d2 = im_refer_512_lin_0.size()
        im_refer_512_lin_1 =im_refer_512_lin_0.reshape([a2,-1, d2])
        
        ########################################

        scatch_512_lin_3=self._layer_permut_channel_SA(scatch_512_lin_2)
        m_SA= torch.matmul(  im_512_lin_2,scatch_512_lin_3) 
         
        m_SA=self._layer_SfTMax(m_SA)
        deep_morphing_0=torch.matmul( m_SA,im_refer_512_lin_1) 
        deep_morphing_1= deep_morphing_0.reshape([a2, b2, c2,d2])
        deep_morphing_2=self._layer_permut_channelfirst(deep_morphing_1)
        if 0:
            print('=====================')
            print(im_512_lin_0.shape)
            print(im_512_lin_1.shape)
            print(im_512_lin_2.shape)
            print(scatch_512_lin_0.shape)
            print(scatch_512_lin_1.shape)
            print(scatch_512_lin_2.shape)
            print(scatch_512_lin_3.shape)
            print(m_SA.shape)
            print(deep_morphing_1.shape)
            print('=====================')
         
        
        return deep_morphing_2,m_SA
    def _get_regularizer(self):
        return self.regularizer



############################################################################################################
class  morphing_512_0(Layer_09):
    def __init__(self,  imageSize, last_activate, L1 = 0., L2 = 0.,\
                 cannal_routine = 0,device = None, train_mode=True ):
        super( morphing_512_0, self).__init__(  imageSize, imageSize) 
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.class_name = self.__class__.__name__
        self.last_activate = last_activate
        self.show=0
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            
 
        param = {'imageSize':[1, 256, 512, 1] ,'alpha': 0.01 ,'w_' :[16,16,20,18,18],'dilation':[1,2,5,4,6]}
        self.decomposition_xtxr=im2txtrTnzr03(param,self.device)            
        param1 = {'imageSize':[1, 256, 512, 1] ,'alpha': 0.01 ,'w_' :[16,16,20,18,18],'dilation':[1,2,5,4,6]}
        self.decomposition_xtxr_1=im2txtrTnzr03(param1,self.device)            
        param2 = {'imageSize':[1, 256, 512, 1] ,'alpha': 0.01 ,'w_' :[16 ],'dilation':[1 ]}
        self.decomposition_xtxr_2=im2txtrTnzr03(param2,self.device)            
            
 
        for param in self.decomposition_xtxr.parameters():
            #print(param.shape)
            param.requires_grad =False
        for param in self.decomposition_xtxr_1.parameters():
            #print(param.shape)
            param.requires_grad =False
        for param in self.decomposition_xtxr_2.parameters():
            #print(param.shape)
            param.requires_grad =False
 
            
        self.MSELoss=nn.MSELoss(reduction='mean')    
        self.imageSize = imageSize
        self.canal_sc = imageSize[-1]
        self.regularizer = Regularizer(L1, L2)
        self.cannal_routine =  cannal_routine
         
        self.L1=L1
        self.L2=L2
        self.count=0
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))

 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.mul_SA_modul_16_1=mul_SA_02([256,82,64,32],[256,82,64,32],self.L1, self.L2, self.device) 
        self.mul_SA_modul_16_2=mul_SA_02([256,84,62,32],[256,82,64,32],self.L1, self.L2, self.device) 
        self.mul_SA_modul_20_5=mul_SA_02([400,128,82,64],[400,256,128,64],self.L1, self.L2, self.device) 
        self.mul_SA_modul_18_4=mul_SA_02([324,128,82,64],[324,256,128,64],self.L1, self.L2, self.device) 
        self.mul_SA_modul_18_6=mul_SA_02([324,128,82,64],[324,256,128,64],self.L1, self.L2, self.device) 
        
        if ~train_mode:
            self.mul_SA_modul_16_1.load_state_dict(torch.load('mul_16_1_00.pt', map_location=self.device))
            self.mul_SA_modul_16_2.load_state_dict(torch.load('mul_16_2_00.pt', map_location=self.device))
            self.mul_SA_modul_20_5.load_state_dict(torch.load('mul_20_5_00.pt', map_location=self.device))
            self.mul_SA_modul_18_4.load_state_dict(torch.load('mul_18_4_00.pt', map_location=self.device))
            self.mul_SA_modul_18_6.load_state_dict(torch.load('mul_18_6_00.pt', map_location=self.device))


        
        for param in self.mul_SA_modul_16_1.parameters():             
            param.requires_grad =train_mode
        for param in self.mul_SA_modul_16_2.parameters():             
            param.requires_grad =train_mode
        for param in self.mul_SA_modul_20_5.parameters():             
            param.requires_grad =train_mode
        for param in self.mul_SA_modul_18_4.parameters():             
            param.requires_grad =train_mode
        for param in self.mul_SA_modul_18_6.parameters():             
            param.requires_grad =train_mode
        
        
        
        self.eye_512=torch.eye(512).to(self.device)
         
         
        self.eye_512.requires_grad =False 
         
        
         
        
        
        self.to(self.device)        
        self.reset_parameters()


    ##############################################################3
    ##############################################################
    def forward(self, img_1 , txtr_im  ):
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
            
        x_input = (img_1, txtr_im   )
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        img_1_inp = self._contiguous(_x_input[0])
        scatch = self._contiguous(_x_input[1])
         
         
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        _layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
         
        scatch_ = _layer_permut_channelfirst(scatch)
        im=_layer_permut_channelfirst(img_1_inp)
        im0=self.decomposition_xtxr.get_features_01(im)
        scatch_0=self.decomposition_xtxr.get_features_01(scatch_)    
        
         
        im_16_1=im0['16_1']# torch.Size([2, 256, 16, 32])
        im_16_2=im0['16_2']# torch.Size([2, 256, 16, 32])
        im_20_5=im0['20_5']# torch.Size([2, 400, 16, 32])
        im_18_4=im0['18_4']#  torch.Size([2, 324, 16, 32])
        im_18_6=im0['18_6']# torch.Size([2, 324, 16, 32])
        
        scatch_16_1=scatch_0['16_1']#  torch.Size([2, 256, 16, 32])
        scatch_16_2=scatch_0['16_2']# torch.Size([2, 256, 16, 32])
        scatch_20_5=scatch_0['20_5']#torch.Size([2, 400, 16, 32])
        scatch_18_4=scatch_0['18_4']# torch.Size([2, 324, 16, 32])
        scatch_18_6=scatch_0['18_6']# torch.Size([2, 324, 16, 32])
        
        del(im0,scatch_0)
        
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        deep_morphing_16_1, self.SA_16_1=self.mul_SA_modul_16_1(im_16_1,scatch_16_1,scatch_16_1)# torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        deep_morphing_16_2, self.SA_16_2=self.mul_SA_modul_16_2(im_16_2,scatch_16_2,scatch_16_1)#  torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        deep_morphing_20_5, self.SA_20_5=self.mul_SA_modul_20_5(im_20_5,scatch_20_5,scatch_16_1)# torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        deep_morphing_18_4, self.SA_18_4=self.mul_SA_modul_18_4(im_18_4,scatch_18_4,scatch_16_1)# torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        deep_morphing_18_6, self.SA_18_6=self.mul_SA_modul_18_6(im_18_6,scatch_18_6,scatch_16_1)#  torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        if 0:
            print(im_16_1.shape,im_16_2.shape,im_20_5.shape,im_18_4.shape,im_18_6.shape )
            print(deep_morphing_16_1.shape,deep_morphing_16_2.shape,\
                  deep_morphing_20_5.shape,deep_morphing_18_4.shape,deep_morphing_18_6.shape )
            print(self.SA_16_1.shape,self.SA_16_2.shape,\
                  self.SA_20_5.shape,self.SA_18_4.shape,self.SA_18_6.shape )

        x=(1/27)*deep_morphing_16_1+(2/27)*deep_morphing_16_2+\
                (10/27)*deep_morphing_20_5+(6/27)*deep_morphing_18_4+(8/27)*deep_morphing_18_6
        ###################    

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
         
        return x
    
    
        ##############################################################
    def morphy_00(self, img_1 , txtr_im  ):
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
            
        x_input = (img_1, txtr_im   )
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        img_1_inp = self._contiguous(_x_input[0])
        scatch = self._contiguous(_x_input[1])
         
         
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        _layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
         
        scatch_ = _layer_permut_channelfirst(scatch)
        im=_layer_permut_channelfirst(img_1_inp)
        im0=self.decomposition_xtxr.get_features_01(im)
        scatch_0=self.decomposition_xtxr.get_features_01(scatch_)    
        
         
        im_16_1=im0['16_1']# torch.Size([2, 256, 16, 32])
        im_16_2=im0['16_2']# torch.Size([2, 256, 16, 32])
        im_20_5=im0['20_5']# torch.Size([2, 400, 16, 32])
        im_18_4=im0['18_4']#  torch.Size([2, 324, 16, 32])
        im_18_6=im0['18_6']# torch.Size([2, 324, 16, 32])
        
        scatch_16_1=scatch_0['16_1']#  torch.Size([2, 256, 16, 32])
        scatch_16_2=scatch_0['16_2']# torch.Size([2, 256, 16, 32])
        scatch_20_5=scatch_0['20_5']#torch.Size([2, 400, 16, 32])
        scatch_18_4=scatch_0['18_4']# torch.Size([2, 324, 16, 32])
        scatch_18_6=scatch_0['18_6']# torch.Size([2, 324, 16, 32])
        
        del(im0,scatch_0)
        
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        deep_morphing_16_1, _=self.mul_SA_modul_16_1(im_16_1,scatch_16_1,scatch_16_1)# torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        deep_morphing_16_2, _ =self.mul_SA_modul_16_2(im_16_2,scatch_16_2,scatch_16_1)#  torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        deep_morphing_20_5, _=self.mul_SA_modul_20_5(im_20_5,scatch_20_5,scatch_16_1)# torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        deep_morphing_18_4, _ =self.mul_SA_modul_18_4(im_18_4,scatch_18_4,scatch_16_1)# torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        deep_morphing_18_6, _ =self.mul_SA_modul_18_6(im_18_6,scatch_18_6,scatch_16_1)#  torch.Size([2, 256, 16, 32]) torch.Size([2, 512, 512])
        if 0:
            print(im_16_1.shape,im_16_2.shape,im_20_5.shape,im_18_4.shape,im_18_6.shape )
            print(deep_morphing_16_1.shape,deep_morphing_16_2.shape,\
                  deep_morphing_20_5.shape,deep_morphing_18_4.shape,deep_morphing_18_6.shape )
            print(self.SA_16_1.shape,self.SA_16_2.shape,\
                  self.SA_20_5.shape,self.SA_18_4.shape,self.SA_18_6.shape )

        x=(1/27)*deep_morphing_16_1+(2/27)*deep_morphing_16_2+\
                (10/27)*deep_morphing_20_5+(6/27)*deep_morphing_18_4+(8/27)*deep_morphing_18_6
         
        return x

    #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    def _get_regularizer(self):
        return self.regularizer

    ###################################################################################################
    def loss_decomposition_01(self,x_,y_):
        y_0 = self._layer_permut_channelfirst(y_)
        #print('444444444444444444444444444')  
        #print('y_0',y_0.shape)  
        y_decomposed=self.decomposition_xtxr_2.get_features_01(y_0)
        y_1=y_decomposed['16_1']
        if 0:
            print('===============================')  
            print('x_',x_.shape) 
            print('y_1', y_1.shape) 
            print('===============================') 
         
            ai_2(np.mean(x_.detach().numpy()[0,:,:,:],0))
            print('predict')
            ai_2(np.mean(y_1.detach().numpy()[0,:,:,:],0))
            print('target')
       
        return self.MSELoss( x_,  y_1)
    
    def loss_batch_02(self,loss_func, xb, yb,   opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


        pred = self(*xb)
        
        if isinstance(pred, tuple):
            pred0 = pred 
             
            del pred
        else:
            pred0 = pred
        loss=0
        loss_glob_1 = self.loss_decomposition_01(pred0, yb)
         
         
        l_SA_16_1=self.MSELoss( self.eye_512,  self.SA_16_1)
        l_SA_16_2=self.MSELoss( self.eye_512,  self.SA_16_2)
        l_SA_20_5=self.MSELoss( self.eye_512,  self.SA_20_5)
        l_SA_18_4=self.MSELoss( self.eye_512,  self.SA_18_4)
        l_SA_18_6=self.MSELoss( self.eye_512,  self.SA_18_6)
        
         
        #print('loss_glob',loss_glob)
        loss+=(5.01*loss_glob_1 + 3.5*l_SA_16_1+3.1*l_SA_16_2+3.5*l_SA_20_5+4.5*l_SA_18_4+7.5*l_SA_18_6)
        #print(loss_glob,loss_mse ,loss_txtr)
        

        
        #loss+=  2.1*loss_mse 
        #loss = loss_func(pred0, yb)
        #loss = cross_entropy(pred, yb)

        del( pred0,self.SA_16_1,self.SA_16_2,self.SA_20_5,self.SA_18_4,self.SA_18_6)

        #_, predicted = torch.max(pred.data, dim = 1)
        #_, ind_target = torch.max(yb, dim = 1)
        #correct = (predicted == ind_target).sum().item()
        #acc = correct / len(yb) #.size(0)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if opt is not None:
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss

        return loss_item, len(yb)#, acc
    ###################################################################################################
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№    
    def fit_SA_512(self, dscrm_model,loader,   epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

        #        _criterion = nn.MSELoss(reduction='mean')
        #        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        if self.cannal_routine==0:
            cannal_=0
        else:    
            cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   


        history = History()
        self.count=0
        for epoch in range(epochs):
            self._optimizer.zero_grad()

            print("Epoch {0}/{1}".format(epoch, epochs), end='')

            self.train()
            ########################################3


            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            for s in loader:
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['images_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['sketch_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['images_ref'].numpy()).to(self.device)
                                        )





                images_cur=train_ds.tensors[0] 
                sketch_cur=train_ds.tensors[1]
                images_ref=train_ds.tensors[2]

                ###ai_2(model_SRR_predict[0,:,:,cannal_ ].cpu().detach().numpy())

                losses_, nums_   =   self.loss_batch_02(dscrm_model, \
                                                   (sketch_cur, images_ref ),\
                                                   images_cur,   self._optimizer)                                                                                                       


                losses.append(losses_)
                nums.append(nums_ )


            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            ######################################

            ### test mode ###
            if validation_loader is not None:
                if self.cannal_routine==0:
                    cannal_=0
                else:    
                    cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   


                self.eval()


                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))

                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['images_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['sketch_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['images_ref'].numpy()).to(self.device)
                                        )





                    images_cur=val_ds.tensors[0] 
                    sketch_cur=val_ds.tensors[1]
                    images_ref=val_ds.tensors[2]








                    losses_, nums_   =  \
                    self.loss_batch_02( dscrm_model,\
                           (sketch_cur, images_ref ),\
                           images_cur, self._optimizer)      

                    losses.append(losses_)
                    nums.append(nums_ )
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                    #acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})

                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')

            print("")


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№    

############################################################################################################
class  morphing_128_0(Layer_09):
    def __init__(self,  imageSize, last_activate, L1 = 0., L2 = 0.,\
                 cannal_routine = 0,device = None ,train_mode=True):
        super( morphing_128_0, self).__init__(  imageSize, imageSize) 
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.class_name = self.__class__.__name__
        self.last_activate = last_activate
        self.show=0
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            
 
        param = {'imageSize':[1, 256, 512, 1] ,'alpha': 0.01 ,'w_' :[32,32,32],'dilation':[1,2,3]}
        self.decomposition_xtxr=im2txtrTnzr03(param,self.device)            
        param1 = {'imageSize':[1, 256, 512, 1] ,'alpha': 0.01 ,'w_' :[32,32,32],'dilation':[1,2,3]}
        self.decomposition_xtxr_1=im2txtrTnzr03(param1,self.device)            
        param2 = {'imageSize':[1, 256, 512, 1] ,'alpha': 0.01 ,'w_' :[32 ],'dilation':[1 ]}
        self.decomposition_xtxr_2=im2txtrTnzr03(param2,self.device)            
            
 
        for param in self.decomposition_xtxr.parameters():
            #print(param.shape)
            param.requires_grad =False
        for param in self.decomposition_xtxr_1.parameters():
            #print(param.shape)
            param.requires_grad =False
        for param in self.decomposition_xtxr_2.parameters():
            #print(param.shape)
            param.requires_grad =False
 
            
        self.MSELoss=nn.MSELoss(reduction='mean')    
        self.imageSize = imageSize
        self.canal_sc = imageSize[-1]
        self.regularizer = Regularizer(L1, L2)
        self.cannal_routine =  cannal_routine
         
        self.L1=L1
        self.L2=L2
        self.count=0
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))

 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.mul_SA_modul_32_1=mul_SA_02([512,256,128,32],[512,256,128,32],self.L1, self.L2, self.device) 
        self.mul_SA_modul_32_2=mul_SA_02([512,256,128,32],[512,256,128,32],self.L1, self.L2, self.device) 
        self.mul_SA_modul_32_3=mul_SA_02([512,256,128,32],[512,256,128,32],self.L1, self.L2, self.device) 
         
        if ~train_mode:
            self.mul_SA_modul_32_1.load_state_dict(torch.load('mul_32_1_00.pt', map_location=self.device))
            self.mul_SA_modul_32_2.load_state_dict(torch.load('mul_32_2_00.pt', map_location=self.device))
            self.mul_SA_modul_32_3.load_state_dict(torch.load('mul_32_3_00.pt', map_location=self.device))
 


        
        for param in self.mul_SA_modul_32_1.parameters():             
            param.requires_grad =train_mode
        for param in self.mul_SA_modul_32_2.parameters():             
            param.requires_grad =train_mode
        for param in self.mul_SA_modul_32_3.parameters():             
            param.requires_grad =train_mode
 
        
        
        
        self.eye_128=torch.eye(128).to(self.device)
         
         
        self.eye_128.requires_grad =False 
         
        
         
        
        
        self.to(self.device)        
        self.reset_parameters()


    ##############################################################3
    ##############################################################
    def forward(self, img_1 , txtr_im  ):
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
            
        x_input = (img_1, txtr_im   )
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        img_1_inp = self._contiguous(_x_input[0])
        scatch = self._contiguous(_x_input[1])
         
         
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        _layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
         
        scatch_ = _layer_permut_channelfirst(scatch)
        im=_layer_permut_channelfirst(img_1_inp)
        im0=self.decomposition_xtxr.get_features_01(im)
        scatch_0=self.decomposition_xtxr.get_features_01(scatch_)    
        
         
        im_32_1=im0['32_1']# torch.Size([2, 512, 8, 16])
        im_32_2=im0['32_2']# torch.Size([2, 512, 8, 16])
        im_32_3=im0['32_3']#torch.Size([2, 512, 8, 16])
 
        
        scatch_32_1=scatch_0['32_1']# torch.Size([2, 512, 8, 16])
        scatch_32_2=scatch_0['32_2']# torch.Size([2, 512, 8, 16])
        scatch_32_3=scatch_0['32_3']#torch.Size([2, 512, 8, 16])
 
        
        del(im0,scatch_0)
        
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        deep_morphing_32_1, self.SA_32_1=self.mul_SA_modul_32_1(im_32_1,scatch_32_1,scatch_32_1)# torch.Size([2, 512, 8, 16]) torch.Size([2, 128, 128])
        deep_morphing_32_2, self.SA_32_2=self.mul_SA_modul_32_2(im_32_2,scatch_32_2,scatch_32_1)# torch.Size([2, 512, 8, 16]) torch.Size([2, 128, 128])
        deep_morphing_32_3, self.SA_32_3=self.mul_SA_modul_32_3(im_32_3,scatch_32_3,scatch_32_1)# torch.Size([2, 512, 8, 16]) torch.Size([2, 128, 128])

        if 0:
            print(im_32_1.shape,im_32_2.shape,im_32_3.shape )
            print(deep_morphing_32_1.shape,deep_morphing_32_2.shape,\
                  deep_morphing_32_3.shape)
            print(self.SA_32_1.shape,self.SA_32_2.shape,\
                  self.SA_32_3.shape )

        x=(1/3)*deep_morphing_32_1+(1/3)*deep_morphing_32_2+\
                (1/3)*deep_morphing_32_3 
        ###################    

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
         
        return x
    #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    def _get_regularizer(self):
        return self.regularizer
    
    def morphy_00(self, img_1 , txtr_im  ):
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
            
        x_input = (img_1, txtr_im   )
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        img_1_inp = self._contiguous(_x_input[0])
        scatch = self._contiguous(_x_input[1])
         
         
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        _layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
         
        scatch_ = _layer_permut_channelfirst(scatch)
        im=_layer_permut_channelfirst(img_1_inp)
        im0=self.decomposition_xtxr.get_features_01(im)
        scatch_0=self.decomposition_xtxr.get_features_01(scatch_)    
        
         
        im_32_1=im0['32_1']# torch.Size([2, 512, 8, 16])
        im_32_2=im0['32_2']# torch.Size([2, 512, 8, 16])
        im_32_3=im0['32_3']#torch.Size([2, 512, 8, 16])
 
        
        scatch_32_1=scatch_0['32_1']# torch.Size([2, 512, 8, 16])
        scatch_32_2=scatch_0['32_2']# torch.Size([2, 512, 8, 16])
        scatch_32_3=scatch_0['32_3']#torch.Size([2, 512, 8, 16])
 
        
        del(im0,scatch_0)
        
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        deep_morphing_32_1, _ =self.mul_SA_modul_32_1(im_32_1,scatch_32_1,scatch_32_1)# torch.Size([2, 512, 8, 16]) torch.Size([2, 128, 128])
        deep_morphing_32_2, _=self.mul_SA_modul_32_2(im_32_2,scatch_32_2,scatch_32_1)# torch.Size([2, 512, 8, 16]) torch.Size([2, 128, 128])
        deep_morphing_32_3, _=self.mul_SA_modul_32_3(im_32_3,scatch_32_3,scatch_32_1)# torch.Size([2, 512, 8, 16]) torch.Size([2, 128, 128])

        if 0:
            print(im_32_1.shape,im_32_2.shape,im_32_3.shape )
            print(deep_morphing_32_1.shape,deep_morphing_32_2.shape,\
                  deep_morphing_32_3.shape)
            print(self.SA_32_1.shape,self.SA_32_2.shape,\
                  self.SA_32_3.shape )

        x=(1/3)*deep_morphing_32_1+(1/3)*deep_morphing_32_2+\
                (1/3)*deep_morphing_32_3 
        
         
        return x

    ###################################################################################################
    def loss_decomposition_01(self,x_,y_):
        y_0 = self._layer_permut_channelfirst(y_)
        #print('444444444444444444444444444')  
        #print('y_0',y_0.shape)  
        y_decomposed=self.decomposition_xtxr_2.get_features_01(y_0)
        y_1=y_decomposed['32_1']
        if 0:
            print('===============================')  
            print('x_',x_.shape) 
            print('y_1', y_1.shape) 
            print('===============================') 
         
            ai_2(np.mean(x_.detach().numpy()[0,:,:,:],0))
            print('predict')
            ai_2(np.mean(y_1.detach().numpy()[0,:,:,:],0))
            print('target')
       
        return self.MSELoss( x_,  y_1)
    
    def loss_batch_02(self,loss_func, xb, yb,   opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


        pred = self(*xb)
        
        if isinstance(pred, tuple):
            pred0 = pred 
             
            del pred
        else:
            pred0 = pred
        loss=0
        loss_glob_1 = self.loss_decomposition_01(pred0, yb)
         
         
        l_SA_32_1=self.MSELoss( self.eye_128,  self.SA_32_1)
        l_SA_32_2=self.MSELoss( self.eye_128,  self.SA_32_2)
        l_SA_32_3=self.MSELoss( self.eye_128,  self.SA_32_3)
        
         
        #print('loss_glob',loss_glob)
        loss+=(5.01*loss_glob_1 + 3.5*l_SA_32_1+3.1*l_SA_32_2+3.5*l_SA_32_3 )
        #print(loss_glob,loss_mse ,loss_txtr)
        

        
        #loss+=  2.1*loss_mse 
        #loss = loss_func(pred0, yb)
        #loss = cross_entropy(pred, yb)

        del( pred0,self.SA_32_1,self.SA_32_2,self.SA_32_3 )

        #_, predicted = torch.max(pred.data, dim = 1)
        #_, ind_target = torch.max(yb, dim = 1)
        #correct = (predicted == ind_target).sum().item()
        #acc = correct / len(yb) #.size(0)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if opt is not None:
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss

        return loss_item, len(yb)#, acc
    ###################################################################################################
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№    
    def fit_SA_128(self, dscrm_model,loader,   epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

        #        _criterion = nn.MSELoss(reduction='mean')
        #        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        if self.cannal_routine==0:
            cannal_=0
        else:    
            cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   


        history = History()
        self.count=0
        for epoch in range(epochs):
            self._optimizer.zero_grad()

            print("Epoch {0}/{1}".format(epoch, epochs), end='')

            self.train()
            ########################################3


            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            for s in loader:
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['images_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['sketch_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['images_ref'].numpy()).to(self.device)
                                        )





                images_cur=train_ds.tensors[0] 
                sketch_cur=train_ds.tensors[1]
                images_ref=train_ds.tensors[2]

                ###ai_2(model_SRR_predict[0,:,:,cannal_ ].cpu().detach().numpy())

                losses_, nums_   =   self.loss_batch_02(dscrm_model, \
                                                   (sketch_cur, images_ref ),\
                                                   images_cur,   self._optimizer)                                                                                                       


                losses.append(losses_)
                nums.append(nums_ )


            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            ######################################

            ### test mode ###
            if validation_loader is not None:
                if self.cannal_routine==0:
                    cannal_=0
                else:    
                    cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   


                self.eval()


                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))

                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['images_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['sketch_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['images_ref'].numpy()).to(self.device)
                                        )





                    images_cur=val_ds.tensors[0] 
                    sketch_cur=val_ds.tensors[1]
                    images_ref=val_ds.tensors[2]








                    losses_, nums_   =  \
                    self.loss_batch_02( dscrm_model,\
                           (sketch_cur, images_ref ),\
                           images_cur, self._optimizer)      

                    losses.append(losses_)
                    nums.append(nums_ )
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                    #acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})

                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')

            print("")


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№    
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
class texture_modul_00(Layer_01):
    def __init__(self,imageSize,   L1 = 0., L2 = 0.,device = None):
        super(texture_modul_00, self).__init__()
        self.imageSize = imageSize
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.L1=L1
        self.L2=L2
         
        self.class_name = self.__class__.__name__        
        self.regularizer = Regularizer(L1, L2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.mul_00=mul_layer_5( 64, 0, self.L1, self.L2, self.device)    
        self.mul_01=mul_layer_5( 100,0, self.L1, self.L2, self.device)
        self.mul_02=mul_layer_6( 36, 0, self.L1, self.L2 , self.device)
        self.mul_03=mul_layer_6( 16, 0, self.L1, self.L2 , self.device)
         
        #self.mul_01=mul_layer_0( 256,3,  L1  , L2  , self.device)
        self.conv_layer_upsample_00=conv_layer_universal_upsample_05(512, 256, 1,  True, self.L1, self.L2, self.device )  
        self.conv_layer_upsample_01=conv_layer_universal_upsample_05(512, 256, 1,  True, self.L1, self.L2, self.device )  
        self.conv_layer_upsample_02=conv_layer_universal_upsample_05(320, 128, 3,  True, self.L1, self.L2, self.device ) 
        self.conv_layer_upsample_03=conv_layer_universal_upsample_05(100, 50, 3,  True, self.L1, self.L2, self.device ) 
        self.conv_layer_upsample_04=conv_layer_universal_upsample_05(128+86+16, 64, 3,  True, self.L1, self.L2, self.device ) 
        self.conv_layer_upsample_05=conv_layer_universal_upsample_05(64, 32, 3,  True, self.L1, self.L2, self.device ) 
        self.add_module('conv_layer_uno_03', conv_layer_universal_uno_05\
                (imageSize[-1]+32 , 32, 1,'relu',3,1,1, self.device)
               )
        self.add_module('conv_layer_uno_04', conv_layer_universal_uno_05\
                (32, 16, 1,'relu',3,1,1, self.device)
               )
        self.add_module('conv_layer_uno_05', conv_layer_universal_uno_05\
                        (16, 8, 1,'relu',3,1,1, self.device)
                       )
        self.add_module('conv_layer_uno_06', conv_layer_universal_uno_05\
                        (8, 4, 1,'relu',3,1,1, self.device)
                       )
         
        self.add_module('conv_layer_uno_07', conv_layer_universal_uno_05\
                        (4, 1, 1,'linear',3,1,1, self.device)
                       )

        self.to(self.device)
        self.reset_parameters()
        
    def forward(self, deep_morphing_512,deep_morphing_256,\
                im_16,scatch_16, im_36,scatch_36, im_64,scatch_64, im_100,scatch_100,im):
        
        
        
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        _layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
         
         
        y_00=self.conv_layer_upsample_00(deep_morphing_512)#torch.Size([1, 256, 16, 32])
         
         
        merged_00=torch.cat((y_00,  deep_morphing_256  ),axis=1)#merged_00 torch.Size([2, 512, 16, 32])
        del(y_00,  deep_morphing_256)
        y_01=self.conv_layer_upsample_00(merged_00)#torch.Size([2, 256, 32, 64])
        mul_64=self.mul_00(im_64,scatch_64) #torch.Size([1, 64, 32, 64]) 
        merged_01=torch.cat((y_01,  mul_64  ),axis=1)#mtorch.Size([2, 320, 32, 64])
        del(y_01,  mul_64 )
        y_02=self.conv_layer_upsample_02(merged_01)#torch.Size([2, 128, 64, 128])
        mul_100=self.mul_01(im_100,scatch_100) #torch.Size([2, 100, 32, 64])

        mul_101=self.conv_layer_upsample_03(mul_100)#torch.Size([2, 50, 64, 128])
        mul_36=self.mul_02(im_36,scatch_36) #torch.Size([2, 36, 64, 128])
        mul_16=self.mul_03(im_16,scatch_16) #torch.Size([2, 16, 64, 128])

        merged_02=torch.cat((y_02,mul_101,  mul_36,mul_16 ),axis=1)#torch.Size([2,128+ 86, 64, 128])
        del(y_02,mul_101,mul_100,  mul_36,mul_16  )
        y_03=self.conv_layer_upsample_04(merged_02)#torch.Size([2, 64, 128, 256])
        y_04=self.conv_layer_upsample_05 (y_03) #torch.Size([2, 32, 256, 512])
        merged_03=torch.cat((y_04,  im ),axis=1)#torch.Size([2, 33, 256, 512])
        del(y_04,  im  )
        alef_3=self._modules['conv_layer_uno_03'](merged_03)
        alef_4=self._modules['conv_layer_uno_04'](alef_3)
        alef_5=self._modules['conv_layer_uno_05'](alef_4)
        alef_6=self._modules['conv_layer_uno_06'](alef_5)
        alef_7=self._modules['conv_layer_uno_07'](alef_6)
        del(merged_03,alef_3,alef_4,alef_5,alef_6  )

        #imageSize     



        x = _layer_permut_channellast(alef_7)
        del(alef_7  ) 
        x = self._contiguous(x)


        return x

         
    def _get_regularizer(self):
        return self.regularizer
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
############################################################################################################
class  model_MUL_21_SRR(Layer_09):
    def __init__(self,  imageSize, last_activate, L1 = 0., L2 = 0.,\
                 cannal_routine = 0,device = None ):
        super( model_MUL_21_SRR, self).__init__(  imageSize, imageSize) 
        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.class_name = self.__class__.__name__
        self.last_activate = last_activate
        self.show=0
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
        param = {'imageSize':[1, 256, 512, 1] ,'alpha': 0.01 ,'w_' :[10,8,6,4]}
        self.decomposition_xtxr=im2txtrTnzr02(param,self.device)
 
        for param in self.decomposition_xtxr.parameters():
            #print(param.shape)
            param.requires_grad =False

 
            
        self.MSELoss=nn.MSELoss(reduction='mean')    
        self.imageSize = imageSize
        self.canal_sc = imageSize[-1]
        self.regularizer = Regularizer(L1, L2)
        self.cannal_routine =  cannal_routine
         
        self.L1=L1
        self.L2=L2
        self.count=0
        self._layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        self._layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))

 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            
            
        self.morphing_128_0 = morphing_128_0\
                    (imageSize = self.imageSize, last_activate='sigmoid', device=self.device,train_mode=False)     
            
        self.morphing_512_0 = morphing_512_0\
                    (imageSize = self.imageSize, last_activate='sigmoid', device=self.device,train_mode=False)




         
        ######################################
        self.texture_modul_=texture_modul_00(imageSize,self.L1, self.L2, self.device) 
        self.texture_modul_.load_state_dict(torch.load('SRR_426_txtr.pt', map_location=self.device))
        #load_state('SRR_426_txtr.pt') 
        for param in self.texture_modul_.parameters():
            #print(param.shape)
            param.requires_grad = True   
        
        self.to(self.device)
        
        self.reset_parameters()


    ##############################################################3
    ##############################################################
    def forward(self, img_1 , txtr_im  ):
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
            
        x_input = (img_1, txtr_im   )
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        img_1_inp = self._contiguous(_x_input[0])
        scatch = self._contiguous(_x_input[1])
        #######################################
  
        
        
        deep_morphing_512=self.morphing_128_0.morphy_00(img_1_inp,scatch)# torch.Size([2, 512, 8, 16])
        #print('self.morphing_128_0.morphy_00(im,scatch_)',deep_morphing_512.shape)
        deep_morphing_256=self.morphing_512_0.morphy_00(img_1_inp,scatch)# torch.Size([2, 256, 16, 32])
        #print('self.morphing_512_0.morphy_00(im,scatch_)',deep_morphing_256.shape)
         
        if 0:
            print('img_1_inp',img_1_inp.shape)
            print('scatch',scatch.shape)
            print('q687',q687.shape)
        ##############        
        
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))
        _layer_permut_channel_SA = Lambda(lambda x:  x.permute((0,2, 1 )))
         
        scatch_ = _layer_permut_channelfirst(scatch)
        im=_layer_permut_channelfirst(img_1_inp)
        #print('im.shape',im.shape)
        #print('scatch_',scatch_.shape)
        #im=self.vgg_normalization(im)
        #print(im.shape)
        #struct=self.vgg_19_decomposition_xtxr.vgg_conv(im)
        #print(struct.shape)self.vgg_normalization
        #########################################################3
        
        
        im0=self.decomposition_xtxr.get_features_01(im)
        #[32,24,16,10,8,6,4]
         
        im_16=im0['4']# torch.Size([2, 16, 64, 128])
        im_36=im0['6']#torch.Size([2, 36, 64, 128])
        im_100=im0['10']# torch.Size([2, 100, 32, 64])
        
        im_64=im0['8']#orch.Size([2, 64, 32, 64])
         
        #torch.Size([2, 16, 64, 128]) torch.Size([2, 36, 64, 128]) torch.Size([2, 100, 32, 64]) torch.Size([2, 576, 16, 32])
        #print(im_16.shape,im_36.shape,im_100.shape,im_576.shape) 
 
        
        scatch_0=self.decomposition_xtxr.get_features_01(scatch_)
        scatch_16=scatch_0['4']#torch.Size([2, 16, 64, 128])
        scatch_36=scatch_0['6']#torch.Size([2, 36, 64, 128])
        scatch_100=scatch_0['10']#torch.Size([2, 100, 32, 64])

        scatch_64=scatch_0['8']#orch.Size([2, 64, 32, 64])
        
        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        x = self.texture_modul_(deep_morphing_512,deep_morphing_256,\
                im_16,scatch_16, im_36,scatch_36, im_64,scatch_64, im_100,scatch_100,im)#orch.Size([2, 256, 512, 1])   
        #print('texture_modul_',x.shape) 
        #print(deep_morphing_256.shape,SA_256.shape)
        ###################    

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
         
        return x
    #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    def _get_regularizer(self):
        return self.regularizer
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    ###################################################################################################
    def loss_batch_00(self,loss_func, xb, yb,   opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


        pred = self(*xb)

        if isinstance(pred, tuple):
            pred0 = pred[0]
            del pred
        else:
            pred0 = pred
        loss=0
        if 0:
            for ghj in range(5):
                i_=int(np.random.rand()*(pred0.shape[1]-self.w_))
                j_=int(np.random.rand()*(pred0.shape[2]-self.w_))
                square=pred0[:,i_:i_+self.w_,   j_  :j_+self.w_,:]
                #print(square.shape)
                #print(loss_func)
                l=loss_func(square,square )[0] 
                loss_local=l**2 
                #print('loss_local',loss_local)
                loss+=1*loss_local
        
          
         

        #loss_glob = self.loss_decomposition_00(pred0, yb)
        if 0:
            loss_glob_1 = self.loss_decomposition_01(pred0, yb)
            loss+=(1.17*loss_glob_1 + 0  )
        else:
            loss_mse = self._criterion(pred0, yb)
            loss+=(0 + 1*loss_mse  )
        #print(loss_glob,loss_mse ,loss_txtr)
        

        
        #loss+=  2.1*loss_mse 
        #loss = loss_func(pred0, yb)
        #loss = cross_entropy(pred, yb)

        del pred0

        #_, predicted = torch.max(pred.data, dim = 1)
        #_, ind_target = torch.max(yb, dim = 1)
        #correct = (predicted == ind_target).sum().item()
        #acc = correct / len(yb) #.size(0)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if opt is not None:
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss

        return loss_item, len(yb)#, acc
################################################################

################################################################
    def fit_dataloader_11(self, dscrm_model,loader,   epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


        #№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
        if self.cannal_routine==0:
            cannal_=0
        else:    
            cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   
            
            
        history = History()
        self.count=0
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            ########################################3
             
            
            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            for s in loader:
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['images_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['sketch_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['images_ref'].numpy()).to(self.device)
                                        )
                
                 
                
                 
                
                images_cur=train_ds.tensors[0] 
                sketch_cur=train_ds.tensors[1]
                images_ref=train_ds.tensors[2]
 
                ###ai_2(model_SRR_predict[0,:,:,cannal_ ].cpu().detach().numpy())

                losses_, nums_   =   self.loss_batch_00(dscrm_model, \
                                                   (sketch_cur, images_ref ),\
                                                   images_cur,   self._optimizer)                                                                                                       
                                    
                                                     
                losses.append(losses_)
                nums.append(nums_ )
                
                
            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            ######################################
             
            ### test mode ###
            if validation_loader is not None:
                if self.cannal_routine==0:
                    cannal_=0
                else:    
                    cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   


                self.eval()
                
                 
                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))
                
                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['images_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['sketch_cur'].numpy()).to(self.device),
                                        torch.FloatTensor(s['images_ref'].numpy()).to(self.device)
                                        )
                
                 
                
                 

                    images_cur=val_ds.tensors[0] 
                    sketch_cur=val_ds.tensors[1]
                    images_ref=val_ds.tensors[2]
                
                

  
                      
                    
                                                                                                                         
               
                    losses_, nums_   =  \
                    self.loss_batch_00( dscrm_model,\
                           (sketch_cur, images_ref ),\
                           images_cur, self._optimizer)      
                                                  
                    losses.append(losses_)
                    nums.append(nums_ )
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                    #acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})
                
                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')
                
            print("")
            
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№    
####################################################################################################################
class Dataset_12_YUV_test(Dataset):
    def __init__(self, video_in ,shape_=[256,512],routine_deform=0, cannals=1):
        files_ = sorted(os.listdir(video_in ))                 
        l_=len(files_)
        #print('len',l_)
        if l_>0:
            file_=files_[0]
            if file_.endswith(".jpg") or file_.endswith(".png") or file_.endswith(".jpeg"):
                files_cur = [video_in+files_[i] for i in range(l_)]
        self.image_file_cur=files_cur
        self.image_file_ref=video_in+file_
        
        
        self.length=l_
        train_part=0.8
        test_part=0.2
        self.length_train = int(self.length * train_part)
        self.length_test = int(self.length * test_part)
        self.shape_=shape_ 
        self.cannals=cannals
        xv, yv = np.meshgrid(range(shape_[0]), range(shape_[1]), sparse=False, indexing='ij')
        xv=2*(xv/shape_[0]-0.5)
        yv=2*(yv /shape_[1]-0.5)
        if routine_deform:
            poly_2d_x=  30*np.sin(apply_poly_00(xv,[0,-4,50,10,5])/15)    
            poly_2d_y=  30*np.cos(apply_poly_00(yv,[0,-27,-28,-39])/15 ) 
        else:# нет деформации
            poly_2d_x=   apply_poly_00(xv,[0,0,0,0,0]) 
            poly_2d_y=   apply_poly_00(yv,[0,0,0,0]) 


        self.mot_xy=1*np.concatenate([np.expand_dims(poly_2d_x,2),np.expand_dims(poly_2d_y,2)],2) 
        if 0:
            show_surf_00(xv, yv, poly_2d_x+poly_2d_y)
        self.multiskotch_=multiskotch_01(bit_per_pixel=2,rotine_sketch=0) 

        
    def set_train_mode(self):
        self.test_mode = False
        self.length = self.length_train
 
    def set_test_mode(self):
        self.test_mode = True
        self.length = self.length_test
 
    def __len__(self):
        return self.length
    ##########################################################################3
    def triplet_TL_00(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 0:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image, (self.shape_[1],self.shape_[0]))
        image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        self.multiskotch_.sketch_multi(image_YUV)
        y_sketch=self.multiskotch_.multi_sketch 
 

        return image_YUV/255,  y_sketch
    def triplet_TL_01(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 1:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image_RGB, (self.shape_[1],self.shape_[0]))
         
        self.multiskotch_.sketch_multi(image)
        y_sketch=self.multiskotch_.multi_sketch 
        image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        return image_YUV/255,  y_sketch
 
    def __getitem__(self, idx):
         
        f_name_cur= self.image_file_cur[idx] 
        f_name_ref= self.image_file_ref
        if 0:
            print(idx)
            print(f_name_cur)
            print(f_name_ref)
            print('-----------------')
         



        
        im_gadol, sketch=self.triplet_TL_01(f_name_cur) 
        im_gadol_ref, sketch_ref=self.triplet_TL_01(f_name_ref) 
        if 0:
            sketch_1= shift_flow_T_01(sketch_ref ,self.mot_xy)
            im_gadol_1=shift_flow_T_01(im_gadol_ref ,self.mot_xy)
        else:
            sketch_1= sketch_ref
            im_gadol_1=im_gadol_ref
            

        sketch01 = sketch  
        
        
        
        sketch_cur=sketch01
        im_gadol_Y=np.expand_dims(im_gadol[:,:,0],2)
        im_gadol_1_Y=np.expand_dims(im_gadol_1[:,:,0],2)
            
            
        cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   
        im_gadol_UV=np.expand_dims(im_gadol[:,:,cannal_],2)
        im_gadol_1_UV=np.expand_dims(im_gadol_1[:,:,cannal_],2)
         
         
 

           

        

         
         
        return {
            'images_cur': im_gadol_Y,             
            'sketch_cur':sketch_cur,
            'images_ref':im_gadol_1_Y,
            'skotch_shift':sketch_1,
            'images_ref_UV': im_gadol_1_UV,
            'images_cur_UV': im_gadol_UV
            
        }
##########################################################
class Dataset_12_YUV_train(Dataset):
    def __init__(self, base0 ,shape_=[256,512], routine_deform=1, cannals=1):
        self.base0 = base0 
        self.length=len(self.base0.df)
        train_part=0.8
        test_part=0.2
        self.length_train = int(self.length * train_part)
        self.length_test = int(self.length * test_part)
        self.shape_=shape_ 
        self.cannals=cannals
        xv, yv = np.meshgrid(range(shape_[0]), range(shape_[1]), sparse=False, indexing='ij')
        xv=2*(xv/shape_[0]-0.5)
        yv=2*(yv /shape_[1]-0.5)
        if routine_deform:
            poly_2d_x=  30*np.sin(apply_poly_00(xv,[0,-4,50,10,5])/15)    
            poly_2d_y=  30*np.cos(apply_poly_00(yv,[0,-27,-28,-39])/15 ) 
        else:# нет деформации
            poly_2d_x=   apply_poly_00(xv,[0,0,0,0,0]) 
            poly_2d_y=   apply_poly_00(yv,[0,0,0,0]) 

        self.routine_deform=routine_deform
        self.mot_xy=1*np.concatenate([np.expand_dims(poly_2d_x,2),np.expand_dims(poly_2d_y,2)],2) 
        if 0:
            show_surf_00(xv, yv, poly_2d_x+poly_2d_y)
        self.multiskotch_=multiskotch_01(bit_per_pixel=2,rotine_sketch=0) 

        
    def set_train_mode(self):
        self.test_mode = False
        self.length = self.length_train
 
    def set_test_mode(self):
        self.test_mode = True
        self.length = self.length_test
 
    def __len__(self):
        return len(self.base0.df)
    ##########################################################################3
    def triplet_TL_00(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 0:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image, (self.shape_[1],self.shape_[0]))
        image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        self.multiskotch_.sketch_multi(image_YUV)
        y_sketch=self.multiskotch_.multi_sketch 
 

        return image_YUV/255,  y_sketch
    def triplet_TL_01(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 1:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image_RGB, (self.shape_[1],self.shape_[0]))
         
        self.multiskotch_.sketch_multi(image)
        y_sketch=self.multiskotch_.multi_sketch 
        image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        return image_YUV/255,  y_sketch
 
    def __getitem__(self, idx):
        path=self.base0.path_in
        image_file_cur= self.base0.df.iloc[idx]['cur'] 
        image_file_ref= self.base0.df.iloc[idx]['ref'] 
         
        f_name_cur= os.path.join(path, image_file_cur) 
        f_name_ref= os.path.join(path, image_file_ref) 
        #print(f_name)
        im_gadol, sketch=self.triplet_TL_01(f_name_cur) 
        im_gadol_ref, sketch_ref=self.triplet_TL_01(f_name_ref) 
        if self.routine_deform:
            sketch_1= shift_flow_T_01(sketch_ref ,self.mot_xy)
            im_gadol_1=shift_flow_T_01(im_gadol_ref ,self.mot_xy)
        else:
            sketch_1= sketch_ref
            im_gadol_1=im_gadol_ref
            

        sketch01 = sketch  
 

 
        sketch_cur=sketch01
        im_gadol_Y=np.expand_dims(im_gadol[:,:,0],2)
        im_gadol_1_Y=np.expand_dims(im_gadol_1[:,:,0],2)
            
            
        cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   
        im_gadol_UV=np.expand_dims(im_gadol[:,:,cannal_],2)
        im_gadol_1_UV=np.expand_dims(im_gadol_1[:,:,cannal_],2)
         
        return {
            'images_cur': im_gadol_Y,             
            'sketch_cur': sketch_cur,
            'images_ref': im_gadol_1_Y,
            'skotch_shift': sketch_1,
            'images_ref_UV': im_gadol_1_UV,
            'images_cur_UV': im_gadol_UV
        }
################################################################
####################################################################################################################
class Dataset_12_YUV_test(Dataset):
    def __init__(self, video_in ,shape_=[256,512],routine_deform=0, cannals=1):
        files_ = sorted(os.listdir(video_in ))                 
        l_=len(files_)
        #print('len',l_)
        if l_>0:
            file_=files_[0]
            if file_.endswith(".jpg") or file_.endswith(".png") or file_.endswith(".jpeg"):
                files_cur = [video_in+files_[i] for i in range(l_)]
        self.image_file_cur=files_cur
        self.image_file_ref=video_in+file_
        
        
        self.length=l_
        train_part=0.8
        test_part=0.2
        self.length_train = int(self.length * train_part)
        self.length_test = int(self.length * test_part)
        self.shape_=shape_ 
        self.cannals=cannals
        xv, yv = np.meshgrid(range(shape_[0]), range(shape_[1]), sparse=False, indexing='ij')
        xv=2*(xv/shape_[0]-0.5)
        yv=2*(yv /shape_[1]-0.5)
        if routine_deform:
            poly_2d_x=  30*np.sin(apply_poly_00(xv,[0,-4,50,10,5])/15)    
            poly_2d_y=  30*np.cos(apply_poly_00(yv,[0,-27,-28,-39])/15 ) 
        else:# нет деформации
            poly_2d_x=   apply_poly_00(xv,[0,0,0,0,0]) 
            poly_2d_y=   apply_poly_00(yv,[0,0,0,0]) 


        self.mot_xy=1*np.concatenate([np.expand_dims(poly_2d_x,2),np.expand_dims(poly_2d_y,2)],2) 
        if 0:
            show_surf_00(xv, yv, poly_2d_x+poly_2d_y)
        self.multiskotch_=multiskotch_01(bit_per_pixel=2,rotine_sketch=0) 

        
    def set_train_mode(self):
        self.test_mode = False
        self.length = self.length_train
 
    def set_test_mode(self):
        self.test_mode = True
        self.length = self.length_test
 
    def __len__(self):
        return self.length
    ##########################################################################3
    def triplet_TL_00(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 0:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image, (self.shape_[1],self.shape_[0]))
        image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        self.multiskotch_.sketch_multi(image_YUV)
        y_sketch=self.multiskotch_.multi_sketch 
 

        return image_YUV/255,  y_sketch
    def triplet_TL_01(self,image_file ):
        stream = open(image_file, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray,cv2.IMREAD_COLOR) 
        stream.close()
        if (image is None):
            return None,None



        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 1:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
        image= cv2.resize( image_RGB, (self.shape_[1],self.shape_[0]))
         
        self.multiskotch_.sketch_multi(image)
        y_sketch=self.multiskotch_.multi_sketch 
        image_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        return image_YUV/255,  y_sketch
 
    def __getitem__(self, idx):
         
        f_name_cur= self.image_file_cur[idx] 
        f_name_ref= self.image_file_ref
        if 0:
            print(idx)
            print(f_name_cur)
            print(f_name_ref)
            print('-----------------')
         



        
        im_gadol, sketch=self.triplet_TL_01(f_name_cur) 
        im_gadol_ref, sketch_ref=self.triplet_TL_01(f_name_ref) 
        if 0:
            sketch_1= shift_flow_T_01(sketch_ref ,self.mot_xy)
            im_gadol_1=shift_flow_T_01(im_gadol_ref ,self.mot_xy)
        else:
            sketch_1= sketch_ref
            im_gadol_1=im_gadol_ref
            

        sketch01 = sketch  
        
        
        
        sketch_cur=sketch01
        im_gadol_Y=np.expand_dims(im_gadol[:,:,0],2)
        im_gadol_1_Y=np.expand_dims(im_gadol_1[:,:,0],2)
            
            
        cannal_=int(1.5+np.sign(np.random.rand()-0.5)/2)   
        im_gadol_UV=np.expand_dims(im_gadol[:,:,cannal_],2)
        im_gadol_1_UV=np.expand_dims(im_gadol_1[:,:,cannal_],2)
         
         
 

           

        

         
         
        return {
            'images_cur': im_gadol_Y,             
            'sketch_cur':sketch_cur,
            'images_ref':im_gadol_1_Y,
            'skotch_shift':sketch_1,
            'images_ref_UV': im_gadol_1_UV,
            'images_cur_UV': im_gadol_UV
            
        }
##########################################################
################################################3
#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
class TL_001_mehanit(Layer_06):
    def __init__(self, imageSize,  last_activate, L1 = 0., L2 = 0.,device = None,numclasses=10 ):
        super(TL_001_mehanit, self).__init__( (imageSize[0],imageSize[1],1),imageSize   )    

        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.class_name = self.__class__.__name__
        self.last_activate = last_activate
        self.cannal_in= imageSize[2]
         
        self.imageSize = imageSize
        self.regularizer = Regularizer(L1, L2)
        self.show=0 
        self.L1=L1
        self.L2=L2
        self.numclasses=numclasses 
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ##############3
         
        
 
       
        self.atrous_pyramid_01 = atrous_pyramid_00\
        (32 , 32 ,12, self.L1, self.L2, self.device,self.show)
        self.atrous_pyramid_02 = atrous_pyramid_00\
        (128 , 64 ,32, self.L1, self.L2, self.device,self.show)
 
        ###############
        self.add_module('fully_conn_layer_universal_00',\
                        fully_conn_layer_universal_00([256,128,128,128,128,128],'lin', self.device))
        
        self.add_module('fully_conn_layer_universal_01',\
                        fully_conn_layer_universal_00([128,64,32,16,8,3],'lin', self.device))
        
        ####################
        self.conv_layer_universal_01_downsampl=conv_layer_downsample_01(1, 8,    True, self.L1, self.L2, self.device ) 
        self.conv_layer_universal_02_downsampl=conv_layer_downsample_01(8, 16,   True, self.L1, self.L2, self.device )
        self.conv_layer_universal_03_downsampl=conv_layer_downsample_01(16, 32,   True, self.L1, self.L2, self.device )
        self.conv_layer_universal_04_downsampl=conv_layer_downsample_01(32, 64,   True, self.L1, self.L2, self.device )
        self.conv_layer_universal_05_downsampl=conv_layer_downsample_01(64, 128,   True, self.L1, self.L2, self.device )
        self.conv_layer_universal_06_downsampl=conv_layer_downsample_01(128, 128,   True, self.L1, self.L2, self.device )
        self.conv_layer_universal_07_downsampl=conv_layer_downsample_01(128, 256,   True, self.L1, self.L2, self.device )
        self.add_module('fltn_1', Flatten( ))
        
         
         
          
        _layer_D01 = Linear(128, self.numclasses, bias = True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.01)
        self.add_module('Dropout01', _layer_Dropout01) 
        
        
        
        
        
        _layer_SfTMax = Softmax(dim = -1)
        self.add_module('SfTMax', _layer_SfTMax)  
        _layer_Sgmd = Sigmoid()
        self.add_module('Sgmd', _layer_Sgmd)  
 
        self.to(self.device)
        
        self.reset_parameters()
 
    def forward_eshar_00(self,scatch , im_wire ):
                                                 
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
        
           
        x_input = (scatch , im_wire)
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        scatch = self._contiguous(_x_input[0])
        im_wire = self._contiguous(_x_input[1])
         
         
        ##############
        
            
        
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))

        scatch0 = _layer_permut_channelfirst(scatch)
        
        if self.show:
            print('scatch0',scatch0.shape)
        im_01_dwnsmpl=self.conv_layer_universal_01_downsampl(scatch0)
        if self.show:
            print('im_01_dwnsmpl',im_01_dwnsmpl.shape)    
        im_02_dwnsmpl=self.conv_layer_universal_02_downsampl(im_01_dwnsmpl)
        if self.show:
            print('im_02_dwnsmpl',im_02_dwnsmpl.shape)
        
        
        
         
        ###################
        if 0:
            im_0=self.atrous_pyramid_01(im_02_dwnsmpl )
            if self.show:
                print('atrous_pyramid_01 im_0',im_0.shape)
        else:
            im_0=im_02_dwnsmpl

        im_03_dwnsmpl=self.conv_layer_universal_03_downsampl(im_0)
        if self.show:
            print('im_03_dwnsmpl',im_03_dwnsmpl.shape)
        im_04_dwnsmpl=self.conv_layer_universal_04_downsampl(im_03_dwnsmpl)
        if self.show:
            print('im_04_dwnsmpl',im_04_dwnsmpl.shape)
        if 0:    
            im_04= self.atrous_pyramid_02(im_04_dwnsmpl)
            if self.show:
                print('atrous_pyramid_02 im_04',im_04.shape)
        else:
            im_04=  im_04_dwnsmpl 

        im_05_dwnsmpl=self.conv_layer_universal_05_downsampl(im_04)
        if self.show:
            print('im_05_dwnsmpl',im_05_dwnsmpl.shape)
        im_06_dwnsmpl=self.conv_layer_universal_06_downsampl(im_05_dwnsmpl)
        if self.show:
            print('im_06_dwnsmpl',im_06_dwnsmpl.shape)
        im_07_dwnsmpl=self.conv_layer_universal_07_downsampl(im_06_dwnsmpl)
        if self.show:
            print('im_07_dwnsmpl',im_07_dwnsmpl.shape)
        im_07=self.fltn_1(im_07_dwnsmpl)
        if self.show:
            print('im_07 ',im_07.shape)
        
        im_10=self.fully_conn_layer_universal_00(im_07)
        if self.show:
            print('im_10 ',im_10.shape)

         
        x = im_10
        x = self._contiguous(x)

           

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()
        im_11=self.fully_conn_layer_universal_01(im_10)
        if self.show:
            print('im_11 ',im_11.shape)                
                
        y = im_11
        y = self._contiguous(y)

           

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                y = y.cpu().detach().numpy()
            else:
                y = y.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                y = y.cpu().detach().numpy().tolist()
            else:
                y = y.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return x,y

 
        
    #####################################################
    def forward(self, scatch, im_wire):
        # skotch_N_global, Ref_global,sketch_ref
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
        
           
        x_input = (scatch , im_wire)
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        scatch = self._contiguous(_x_input[0])
        im_wire = self._contiguous(_x_input[1])
         
         
        ##############
        
            
        
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))

        scatch0 = _layer_permut_channelfirst(scatch)
        
        if self.show:
            print('scatch0',scatch0.shape)
        im_01_dwnsmpl=self.conv_layer_universal_01_downsampl(scatch0)
        if self.show:
            print('im_01_dwnsmpl',im_01_dwnsmpl.shape)    
        im_02_dwnsmpl=self.conv_layer_universal_02_downsampl(im_01_dwnsmpl)
        if self.show:
            print('im_02_dwnsmpl',im_02_dwnsmpl.shape)
        
        
        
         
        ###################
        if 0:
            im_0=self.atrous_pyramid_01(im_02_dwnsmpl )
            if self.show:
                print('atrous_pyramid_01 im_0',im_0.shape)
        else:
            im_0=im_02_dwnsmpl

        im_03_dwnsmpl=self.conv_layer_universal_03_downsampl(im_0)
        if self.show:
            print('im_03_dwnsmpl',im_03_dwnsmpl.shape)
        im_04_dwnsmpl=self.conv_layer_universal_04_downsampl(im_03_dwnsmpl)
        if self.show:
            print('im_04_dwnsmpl',im_04_dwnsmpl.shape)
        if 0:    
            im_04= self.atrous_pyramid_02(im_04_dwnsmpl)
            if self.show:
                print('atrous_pyramid_02 im_04',im_04.shape)
        else:
            im_04=  im_04_dwnsmpl 

        im_05_dwnsmpl=self.conv_layer_universal_05_downsampl(im_04)
        if self.show:
            print('im_05_dwnsmpl',im_05_dwnsmpl.shape)
        im_06_dwnsmpl=self.conv_layer_universal_06_downsampl(im_05_dwnsmpl)
        if self.show:
            print('im_06_dwnsmpl',im_06_dwnsmpl.shape)
        im_07_dwnsmpl=self.conv_layer_universal_07_downsampl(im_06_dwnsmpl)
        if self.show:
            print('im_07_dwnsmpl',im_07_dwnsmpl.shape)
        im_07=self.fltn_1(im_07_dwnsmpl)
        if self.show:
            print('im_07 ',im_07.shape)
        
        im_10=self.fully_conn_layer_universal_00(im_07)
        if self.show:
            print('im_10 ',im_10.shape)
        im_11=self.fully_conn_layer_universal_01(im_10)
        if self.show:
            print('im_11 ',im_11.shape)
        
        
        #print('im_10 ',im_10.shape) 
        im_12=self.D01(im_10)
        #print('im_12 ',im_12.shape)
         
        im_12=self.Dropout01(im_12)
        im_13=self.SfTMax(im_12)
        
        
        
        
         
        
        
        
        x = im_13
        x = self._contiguous(x)

        ###################    

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return x
     
    def _get_regularizer(self):
        return self.regularizer
    
     
     ###################################################################################################
    def loss_batch_00(self,dsrmn_model,  x,   opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
        if 0:
            xb=(x[0],x[0]) 
            pred_ancor_0,pred_ancor_1 = self.forward_eshar_00(*xb)
            xb=(x[1],x[1])
            pred_positive_0,pred_positive_1 = self.forward_eshar_00(*xb)
            xb=(x[2],x[2])
            pred_neg_0,pred_neg_1 = self.forward_eshar_00(*xb)
            #print('self._criterion',self._criterion)
            loss_positive_0=self._criterion( pred_ancor_0,pred_positive_0) 
            loss_negative_0=self._criterion( pred_ancor_0,pred_neg_0) 
            loss_positive_1=self._criterion( pred_ancor_1,pred_positive_1) 
            loss_negative_1=self._criterion( pred_ancor_1,pred_neg_1) 
            print(loss_positive_0,loss_negative_0)#,end='')
            loss_0=0.00*loss_positive_0+1.15* (1-loss_negative_0)**2
            loss_1=0.00*loss_positive_1+1.11* (1-loss_negative_1)**2
            print('loss_0',loss_0)#,end='')    
            criterion_tml = torch.nn.TripletMarginLoss(margin=1.0, p=2)
            loss_2=criterion_tml (pred_ancor_0,pred_positive_0,pred_neg_0)
            print('loss_2',loss_2)
        else:
            if 0:
                Positive=x[0][0].numpy() 
                ai_2(Positive[:,:,0])
                Ancor=x[1][0].numpy() 
                ai_2(Ancor[:,:,0])
                neg=x[2][0].numpy() 
                ai_2(neg[:,:,0])
                print('999999999999999999999999999999999')
            
            
            loss_0=0.00
            loss_1=0.00
            xa=(x[0],x[0]) 
            pred_ancor_0 = self(*xa)
            xb=(x[1],x[1])
            pred_positive_0  = self(*xb)
            xc=(x[2],x[2])
            pred_neg_0  = self(*xc)
            criterion_tml = torch.nn.TripletMarginLoss(margin=1.0, p=2)
            loss_2=criterion_tml (pred_ancor_0,pred_positive_0,pred_neg_0)
            #loss_positive_0=self._criterion( pred_ancor_0,pred_positive_0) 
            #loss_negative_0=self._criterion( pred_ancor_0,pred_neg_0)  
            #print('loss_positive_0,loss_negative_', loss_positive_0,loss_negative_0  )
            #print('loss_2',loss_2)
         
        loss =0.0*loss_0+0.0*loss_1+1*loss_2
        
         
        #print(' loss', loss) #,end='')    
       
        #print(self.loss_vgg_1_bw(pred0, yb),self._criterion(pred0, yb),self.MSELoss( dscrm_tenzor,0*dscrm_tenzor))
         
        
        #loss+=  2.1*loss_mse 
        #loss = loss_func(pred0, yb)
        #loss = cross_entropy(pred, yb)

        #del(pred_ancor_0,pred_ancor_1,pred_positive_0,pred_positive_1,pred_neg_0,pred_neg_1) 

        #_, predicted = torch.max(pred.data, dim = 1)
        #_, ind_target = torch.max(yb, dim = 1)
        #correct = (predicted == ind_target).sum().item()
        #acc = correct / len(yb) #.size(0)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if (opt is not None) :
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss
        
        return loss_item, 1#, acc
###################################################################################################
    def loss_batch_02(self,dsrmn_model,  x,   opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
        
        xb=(x[0],x[0]) 
        pred_ancor_0,pred_ancor_1 = self.forward_eshar_00(*xb)
        xb=(x[1],x[1])
        pred_positive_0,pred_positive_1 = self.forward_eshar_00(*xb)
        xb=(x[2],x[2])
        pred_neg_0,pred_neg_1 = self.forward_eshar_00(*xb)
        #print('self._criterion',self._criterion)
        loss_0=self.criterion_tml (pred_ancor_0,pred_positive_0,pred_neg_0)
        loss_1=self.criterion_tml (pred_ancor_1,pred_positive_1,pred_neg_1)
            
            
            
             
            
             
         
        loss =1.0*loss_0+0.9*loss_1 
        
         
        #print(' loss', loss) #,end='')    
       
        #print(self.loss_vgg_1_bw(pred0, yb),self._criterion(pred0, yb),self.MSELoss( dscrm_tenzor,0*dscrm_tenzor))
         
        
        #loss+=  2.1*loss_mse 
        #loss = loss_func(pred0, yb)
        #loss = cross_entropy(pred, yb)

        #del(pred_ancor_0,pred_ancor_1,pred_positive_0,pred_positive_1,pred_neg_0,pred_neg_1) 

        #_, predicted = torch.max(pred.data, dim = 1)
        #_, ind_target = torch.max(yb, dim = 1)
        #correct = (predicted == ind_target).sum().item()
        #acc = correct / len(yb) #.size(0)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if (opt is not None) :
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss
        
        return loss_item, 1#, acc

################################################################
    def loss_batch_01(self,dsrmn_model, xb, yb,   opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

        #print(xb[0].shape)
        pred = self(*xb)
        #print(pred)  
        Positive=xb[0][0].numpy() 
        #ai_2(Positive[:,:,0])
        #print(yb)        
        #print('999999999999999999999999999999999')    

        if isinstance(pred, tuple):
            pred0 = pred[0]
            del pred
        else:
            pred0 = pred
        loss=0
        
        #loss_mse= self._criterion(pred0, yb) 
        #print(pred0.shape)
        #print(yb.shape)
        MSELoss=nn.MSELoss(reduction='mean')    
        loss_mse= MSELoss(pred0, yb) 
         
        
        loss +=1.1*loss_mse 
       
        #print(self.loss_vgg_1_bw(pred0, yb),self._criterion(pred0, yb),self.MSELoss( dscrm_tenzor,0*dscrm_tenzor))
         
        
        #loss+=  2.1*loss_mse 
        #loss = loss_func(pred0, yb)
        #loss = cross_entropy(pred, yb)

        del pred0

        #_, predicted = torch.max(pred.data, dim = 1)
        #_, ind_target = torch.max(yb, dim = 1)
        #correct = (predicted == ind_target).sum().item()
        #acc = correct / len(yb) #.size(0)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if (opt is not None)  :
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss

        return loss_item, len(yb)#, acc

################################################################
    def fit_dataloader_CLASS(self, dscrm_model,loader,   epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


         
            
            
        history = History()
        self.count=0
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            ########################################3
             
            
            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            for s in loader:
                
 
                
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['class_'].numpy()).to(self.device)  
                                        )
                
                 
                
                 
                
                images_Anchor=train_ds.tensors[0] 
                 
                class_=train_ds.tensors[1]
                 

                
                
                

                losses_, nums_   =   self.loss_batch_01(dscrm_model, \
                                                   (images_Anchor ,images_Anchor ),\
                                                    class_,  self._optimizer)                                                                                                       


                losses.append(losses_)
                nums.append(nums_ )
                 
                
            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            ######################################
             
            ### test mode ###
            if validation_loader is not None:
                 


                self.eval()
                
                 
                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))
                
                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['class_'].numpy()).to(self.device)  
 
                                        )
                

                    images_Anchor=val_ds.tensors[0] 

                    class_=val_ds.tensors[1]
                     
                
                 
                
                     
                
                

  
                      
                    
                                                                                                                         

                    losses_, nums_   =  \
                    self.loss_batch_01( dscrm_model,\
                           (images_Anchor ,images_Anchor ),\
                           class_, self._optimizer)      

                    losses.append(losses_)
                    nums.append(nums_ )
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                    #acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})
                
                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')
                
            print("")
            
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history

################################################################
    def fit_dataloader_TL(self, dscrm_model,loader,   epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


         
            
            
        history = History()
        self.count=0
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            ########################################3
             
            
            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            for s in loader:
                
 
                
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['Positive'].numpy()).to(self.device) ,
                                        torch.FloatTensor(s['Negative'].numpy()).to(self.device) 
                                        
                                        )
                
                 
                
                 
                
                images_Anchor=train_ds.tensors[0] 
                 
                images_Positive=train_ds.tensors[1]
                images_Negative=train_ds.tensors[2]

                
                
                

                losses_, nums_   =   self.loss_batch_02(dscrm_model, \
                                                   (images_Anchor ,images_Positive,images_Negative ),\
                                                      self._optimizer)                                                                                                       


                losses.append(losses_)
                nums.append(nums_ )
                 
                
            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            ######################################
             
            ### test mode ###
            if validation_loader is not None:
                 


                self.eval()
                
                 
                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))
                
                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['Positive'].numpy()).to(self.device) ,
                                        torch.FloatTensor(s['Negative'].numpy()).to(self.device) 
                                        
                                        )
                

                    images_Anchor=val_ds.tensors[0] 

                    images_Positive=val_ds.tensors[1]
                    images_Negative=val_ds.tensors[2]
                
                 
                
                     
                
                

  
                      
                    
                                                                                                                         

                    losses_, nums_   =  \
                    self.loss_batch_00( dscrm_model,\
                           (images_Anchor ,images_Positive,images_Negative ),\
                            self._optimizer)      

                    losses.append(losses_)
                    nums.append(nums_ )
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                    #acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})
                
                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')
                
            print("")
            
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
    
###########################################
################################################3
################################################3
class conv_simple_features_00(Layer_06):
    def __init__(self,  device = None, L1 = 0., L2 = 0.,show=0):
        super(conv_simple_features_00, self).__init__()
        self.show = show
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.L1=L1
        self.L2=L2
        self.regularizer = Regularizer(L1, L2)
        _layer_conv_31 = Conv2d(1,4, kernel_size=(5, 5),
                            stride=(4, 4), padding = (2, 2), padding_mode = 'zeros', bias = True)
        self.add_module('conv00', _layer_conv_31) 
        _layer_activation_1 = LeakyReLU(0.05) 
        self.add_module('activation_LeakyReLU', _layer_activation_1)
        _layer_conv_32 = Conv2d(4,16, kernel_size=(5,5),
                            stride=(4, 4), padding = (2, 2), padding_mode = 'zeros', bias = True)
        
        self.add_module('conv01', _layer_conv_32) 
        _layer_conv_33 = Conv2d(16,22, kernel_size=(3,3),
                            stride=(1, 1), padding = (1, 1), padding_mode = 'zeros', bias = True)
          
        
        self.add_module('conv02', _layer_conv_33) 
        _layer_pooling_1 = MaxPool2d(kernel_size=(2, 2))  
        self.add_module('Pool_00', _layer_pooling_1) 
        self.add_module('fltn_1', Flatten( ))

        self.to(self.device)
        self.reset_parameters()
 
    def forward(self, scatch0):
        im_01_dwnsmpl=self.conv00(scatch0)
        im_01_dwnsmpl=self.activation_LeakyReLU(im_01_dwnsmpl)
        if self.show:
            print('im_01_dwnsmpl',im_01_dwnsmpl.shape)    
        im_02_dwnsmpl=self.conv01(im_01_dwnsmpl)
        im_02_dwnsmpl=self.activation_LeakyReLU(im_02_dwnsmpl)
        if self.show:
            print('im_02_dwnsmpl',im_02_dwnsmpl.shape)    
        im_03_dwnsmpl=self.conv02(im_02_dwnsmpl)
        im_03_dwnsmpl=self.Pool_00(im_03_dwnsmpl) 
        im_03_dwnsmpl=self.activation_LeakyReLU(im_03_dwnsmpl)
        if self.show:
            print('im_03_dwnsmpl',im_03_dwnsmpl.shape)    
        vect_00=self.fltn_1(im_03_dwnsmpl)
        if self.show:
            print('vect_00',vect_00.shape)   
        return vect_00
################################################3
class fully_connect_modul_264(Layer_06):
    def __init__(self,  device = None, L1 = 0., L2 = 0., numclasses=9, show=0):
        super(fully_connect_modul_264, self).__init__()
        self.show = show
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.numclasses =numclasses        
        self.L1=L1
        self.L2=L2
        self.regularizer = Regularizer(L1, L2)
        _layer_activation_1 = LeakyReLU(0.05) 
        self.add_module('activation_LeakyReLU', _layer_activation_1)
        _layer_D01 = Linear(352, 256, bias = True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.5)
        _layer_batch_norm_3 = BatchNorm1d(256)
        self.add_module('Dropout01', _layer_Dropout01) 
        self.add_module('layer_batch_norm',  _layer_batch_norm_3) 
        _layer_D02 = Linear(256, 128, bias = True)
        self.add_module('D02', _layer_D02)
        _layer_D03 = Linear(128,  self.numclasses, bias = True)
        self.add_module('D03', _layer_D03)
       
        
        _layer_SfTMax = Softmax(dim = -1)
        self.add_module('SfTMax', _layer_SfTMax)  
        _layer_Sgmd = Sigmoid()
        self.add_module('Sgmd', _layer_Sgmd)  
        #########################
        self.to(self.device)
        self.reset_parameters()
 
    def forward(self, vect_00):
        vect_01=self.D01(vect_00)        
        vect_01=self.Dropout01(vect_01)
        vect_01=self.activation_LeakyReLU(vect_01)
        if self.show:
            print('vect_01',vect_01.shape) 
        vect_01=self.layer_batch_norm(vect_01)
        if self.show:
            print('vect_01 layer_batch_norm',vect_01.shape) 
        vect_02=self.D02(vect_01) 
        vect_02=self.activation_LeakyReLU(vect_02)
        if self.show:
            print('vect_02',vect_02.shape)    
        vect_03=self.D03(vect_02) 
        vect_03=self.SfTMax(vect_03) 
         
        if self.show:
            print('vect_03',vect_03.shape)         
        return vect_03
################################################3
class fully_connect_modul_265(Layer_06):
    def __init__(self,  device = None, L1 = 0., L2 = 0.,   show=0):
        super(fully_connect_modul_265, self).__init__()
        self.show = show
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
             
        self.L1=L1
        self.L2=L2
        self.regularizer = Regularizer(L1, L2)
        _layer_activation_1 = LeakyReLU(0.05) 
        self.add_module('activation_LeakyReLU', _layer_activation_1)
        _layer_D01 = Linear(352, 256, bias = True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.5)
        _layer_batch_norm_3 = BatchNorm1d(256)
        self.add_module('Dropout01', _layer_Dropout01) 
        self.add_module('layer_batch_norm',  _layer_batch_norm_3) 
        _layer_D02 = Linear(256, 128, bias = True)
        self.add_module('D02', _layer_D02)
        _layer_D03 = Linear(128,  64, bias = True)
        self.add_module('D03', _layer_D03)
       
        
        
        _layer_Sgmd = Sigmoid()
        self.add_module('Sgmd', _layer_Sgmd)  
        #########################
        self.to(self.device)
        self.reset_parameters()
 
    def forward(self, vect_00):
        vect_01=self.D01(vect_00)        
        vect_01=self.Dropout01(vect_01)
        vect_01=self.activation_LeakyReLU(vect_01)
        if self.show:
            print('vect_01',vect_01.shape) 
        vect_01=self.layer_batch_norm(vect_01)
        if self.show:
            print('vect_01 layer_batch_norm',vect_01.shape) 
        vect_02=self.D02(vect_01) 
        vect_02=self.activation_LeakyReLU(vect_02)
        if self.show:
            print('vect_02',vect_02.shape)    
        vect_03=self.D03(vect_02) 
        
         
        if self.show:
            print('vect_03',vect_03.shape)         
        return vect_03

#№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
class TL_002_mehanit(Layer_06):
    def __init__(self, imageSize,  last_activate, L1 = 0., L2 = 0.,device = None,numclasses=10,show=0 ):
        super(TL_002_mehanit, self).__init__( (imageSize[0],imageSize[1],1),imageSize   )    

        #self.class_name = str(self.__class__).split(".")[-1].split("'")[0]
        self.class_name = self.__class__.__name__
        self.last_activate = last_activate
        self.cannal_in= imageSize[2]
         
        self.imageSize = imageSize
        self.regularizer = Regularizer(L1, L2)
        self.show=show
        self.L1=L1
        self.L2=L2
        self.numclasses=numclasses 
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        if (device is not None):
            self.device = device if (not isinstance(device, str)) else torch.device(device)
        else:
            self.device = device if (device is not None) else \
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ##############3
        self.conv2Dfeatures=conv_simple_features_00(device,L1 ,L2,show) 
        self.fully_connect_modul_0=fully_connect_modul_264(device,L1 ,L2,numclasses,show) 
        self.fully_connect_modul_TL=fully_connect_modul_265(device,L1 ,L2,show) 
         #######################
        _layer_activation_1 = LeakyReLU(0.05) 
        self.add_module('activation_LeakyReLU', _layer_activation_1)
        _layer_D01 = Linear(352, 256, bias = True)
        self.add_module('D01', _layer_D01)
        _layer_Dropout01 = Dropout(0.5)
        _layer_batch_norm_3 = BatchNorm1d(256)
        self.add_module('Dropout01', _layer_Dropout01) 
        self.add_module('layer_batch_norm',  _layer_batch_norm_3) 
        _layer_D02 = Linear(256, 128, bias = True)
        self.add_module('D02', _layer_D02)
        _layer_D03 = Linear(128,  self.numclasses, bias = True)
        self.add_module('D03', _layer_D03)
        #########################
        
        _layer_SfTMax = Softmax(dim = -1)
        self.add_module('SfTMax', _layer_SfTMax)  
        _layer_Sgmd = Sigmoid()
        self.add_module('Sgmd', _layer_Sgmd)  
 
        self.to(self.device)
        
        self.reset_parameters()
    #####################################################
    def forward(self, scatch, im_wire):
        # skotch_N_global, Ref_global,sketch_ref
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
        
           
        x_input = (scatch , im_wire)
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        scatch = self._contiguous(_x_input[0])
        im_wire = self._contiguous(_x_input[1])
         
         
        ##############
        
            
        
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))

        scatch0 = _layer_permut_channelfirst(scatch)
        vect_00=self.conv2Dfeatures(scatch0)
        vect_03=self.fully_connect_modul_0(vect_00)    
        ################################# 
         
        ######################################
        x = vect_03
        x = self._contiguous(x)

        ###################    

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return x
     
    def _get_regularizer(self):
        return self.regularizer
#################################################################
###################################################################################################
    def loss_batch_02(self,dsrmn_model,  x,   opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
        
        xb=(x[0],x[0]) 
        pred_ancor_0,_ = self.forward_eshar_00(*xb)
        xb=(x[1],x[1])
        pred_positive_0,_ = self.forward_eshar_00(*xb)
        xb=(x[2],x[2])
        pred_neg_0,_ = self.forward_eshar_00(*xb)
        #print('self._criterion',self._criterion)
        loss_0=self.criterion_tml (pred_ancor_0,pred_positive_0,pred_neg_0)
        loss_1=0#self.criterion_tml (pred_ancor_1,pred_positive_1,pred_neg_1)
            
            
            
             
            
             
         
        loss =1.0*loss_0+0.9*loss_1 
        
         
        #print(' loss', loss) #,end='')    
       
        #print(self.loss_vgg_1_bw(pred0, yb),self._criterion(pred0, yb),self.MSELoss( dscrm_tenzor,0*dscrm_tenzor))
         
        
        #loss+=  2.1*loss_mse 
        #loss = loss_func(pred0, yb)
        #loss = cross_entropy(pred, yb)

        #del(pred_ancor_0,pred_ancor_1,pred_positive_0,pred_positive_1,pred_neg_0,pred_neg_1) 

        #_, predicted = torch.max(pred.data, dim = 1)
        #_, ind_target = torch.max(yb, dim = 1)
        #correct = (predicted == ind_target).sum().item()
        #acc = correct / len(yb) #.size(0)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if (opt is not None) :
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss
        
        return loss_item, 1#, acc
################################################################
    #####################################################
    def forward_eshar_00(self, scatch, im_wire):
        # skotch_N_global, Ref_global,sketch_ref
        class _type_input(Enum):
            is_torch_tensor = 0
            is_numpy = 1
            is_list = 2
        
           
        x_input = (scatch , im_wire)
        
        _t_input = []
        _x_input = []
        for x in x_input:
            if isinstance(x, (torch.Tensor)):
                _t_input.append(_type_input.is_torch_tensor)
                _x_input.append(x)
            elif isinstance(x, (np.ndarray)):
                _t_input.append(_type_input.is_numpy)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            elif isinstance(x, (list, tuple)):
                _t_input.append(_type_input.is_list)
                _x_input.append(torch.FloatTensor(x).to(self.device))
            else:
                raise Exception('Invalid type input')

        _x_input = tuple(_x_input)
        _t_input = tuple(_t_input)

        scatch = self._contiguous(_x_input[0])
        im_wire = self._contiguous(_x_input[1])
         
         
        ##############
        
            
        
        _layer_permut_channelfirst = Lambda(lambda x:  x.permute((0, 3, 1, 2)))
        _layer_permut_channellast = Lambda(lambda x:  x.permute((0, 2, 3, 1)))

        scatch0 = _layer_permut_channelfirst(scatch)
        vect_00=self.conv2Dfeatures(scatch0)
        vect_03=self.fully_connect_modul_TL(vect_00)    
        ################################# 
         
        ######################################
        x = vect_03
        x = self._contiguous(x)

        ###################    

        if _type_input.is_torch_tensor in _t_input:
            pass
        elif _type_input.is_numpy in _t_input:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy()
            else:
                x = x.detach().numpy()
        else:
            if (self.device.type == "cuda"):
                x = x.cpu().detach().numpy().tolist()
            else:
                x = x.detach().numpy().tolist()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return x,0
     
    def _get_regularizer(self):
        return self.regularizer

###################################################################
    def loss_batch_01(self,dsrmn_model, xb, yb,   opt=None):
#            def cross_entropy(pred, soft_targets):
#                return -torch.log(torch.mean(torch.sum(soft_targets * pred, 1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#            def l2_norm_diff(pred, soft_targets):
#                return  torch.sqrt(torch.mean(torch.sum((soft_targets - pred )**2,-1)))
            #logsoftmax = nn.LogSoftmax()
            #return torch.pow(1 - torch.mean(torch.sum(soft_targets * pred, 1)), 2)
            #return torch.mean(torch.sum(- soft_targets * torch.norm(pred, p=2,dim=1), 1))
            #return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

        #print(xb[0].shape)
        pred = self(*xb)
        #print(pred)  
        Positive=xb[0][0].numpy() 
        #ai_2(Positive[:,:,0])
        #print(yb)        
        #print('999999999999999999999999999999999')    

        if isinstance(pred, tuple):
            pred0 = pred[0]
            del pred
        else:
            pred0 = pred
        loss=0
        
        #loss_mse= self._criterion(pred0, yb) 
        #print(pred0.shape)
        #print(yb.shape)
        MSELoss=nn.MSELoss(reduction='mean')    
        loss_mse= MSELoss(pred0, yb) 
         
        
        loss +=1.1*loss_mse 
       
        #print(self.loss_vgg_1_bw(pred0, yb),self._criterion(pred0, yb),self.MSELoss( dscrm_tenzor,0*dscrm_tenzor))
         
        
        #loss+=  2.1*loss_mse 
        #loss = loss_func(pred0, yb)
        #loss = cross_entropy(pred, yb)

        del pred0

        #_, predicted = torch.max(pred.data, dim = 1)
        #_, ind_target = torch.max(yb, dim = 1)
        #correct = (predicted == ind_target).sum().item()
        #acc = correct / len(yb) #.size(0)

        _regularizer = self._get_regularizer()

        reg_loss = 0
        for param in self.parameters():
            reg_loss += _regularizer(param)

        loss += reg_loss

        if (opt is not None)  :
            with torch.no_grad():

                opt.zero_grad()

                loss.backward()

                opt.step()

        self.count+=1
        if self.count  %3==0:
            print("*", end='')

        loss_item = loss.item()

        del loss
        del reg_loss

        return loss_item, len(yb)#, acc

    
################################################################
    def fit_dataloader_CLASS(self, dscrm_model,loader,   epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


         
            
            
        history = History()
        self.count=0
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            ########################################3
             
            
            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            for s in loader:
                
 
                
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['class_'].numpy()).to(self.device)  
                                        )
                
                 
                
                 
                
                images_Anchor=train_ds.tensors[0] 
                 
                class_=train_ds.tensors[1]
                 

                
                
                

                losses_, nums_   =   self.loss_batch_01(dscrm_model, \
                                                   (images_Anchor ,images_Anchor ),\
                                                    class_,  self._optimizer)                                                                                                       


                losses.append(losses_)
                nums.append(nums_ )
                 
                
            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            ######################################
             
            ### test mode ###
            if validation_loader is not None:
                 


                self.eval()
                
                 
                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))
                
                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['class_'].numpy()).to(self.device)  
 
                                        )
                

                    images_Anchor=val_ds.tensors[0] 

                    class_=val_ds.tensors[1]
                     
                
                 
                
                     
                
                

  
                      
                    
                                                                                                                         

                    losses_, nums_   =  \
                    self.loss_batch_01( dscrm_model,\
                           (images_Anchor ,images_Anchor ),\
                           class_, self._optimizer)      

                    losses.append(losses_)
                    nums.append(nums_ )
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                    #acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})
                
                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')
                
            print("")
            
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
################################################################
    def fit_dataloader_TL(self, dscrm_model,loader,   epochs = 1, validation_loader = None):
        #_criterion = nn.CrossEntropyLoss(reduction='mean')
        #_optimizer = optim.AdamW(self.parameters())
        #_optimizer = optim.Adam(self.parameters(), lr = 0.00001)#, eps=0.0)
        if (self._criterion is None): # or not isinstance(self._criterion, nn._Loss):
            raise Exception("Loss-function is not select!")

        if (self._optimizer is None) or not isinstance(self._optimizer, optim.Optimizer):
            raise Exception("Optimizer is not select!")

#        _criterion = nn.MSELoss(reduction='mean')
#        _optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.2)


         
            
            
        history = History()
        self.count=0
        for epoch in range(epochs):
            self._optimizer.zero_grad()
            
            print("Epoch {0}/{1}".format(epoch, epochs), end='')
            
            self.train()
            ########################################3
             
            
            ### train mode ###
            print("[", end='')
            losses=[]
            nums=[]
            for s in loader:
                
 
                
                train_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['Positive'].numpy()).to(self.device) ,
                                        torch.FloatTensor(s['Negative'].numpy()).to(self.device) 
                                        
                                        )
                
                 
                
                 
                
                images_Anchor=train_ds.tensors[0] 
                 
                images_Positive=train_ds.tensors[1]
                images_Negative=train_ds.tensors[2]

                
                
                

                losses_, nums_   =   self.loss_batch_02(dscrm_model, \
                                                   (images_Anchor ,images_Positive,images_Negative ),\
                                                      self._optimizer)                                                                                                       


                losses.append(losses_)
                nums.append(nums_ )
                 
                
            print("]", end='')


            sum_nums = np.sum(nums)
            loss = np.sum(np.multiply(losses, nums)) / sum_nums
            ######################################
             
            ### test mode ###
            if validation_loader is not None:
                 


                self.eval()
                
                 
                print("[", end='')
                losses=[]
                nums=[]
                for s in validation_loader:
                #s = next(iter(loader))
                
                    val_ds = TensorDataset(                                                            
                                        torch.FloatTensor(s['Anchor'].numpy()).to(self.device),
                                        torch.FloatTensor(s['Positive'].numpy()).to(self.device) ,
                                        torch.FloatTensor(s['Negative'].numpy()).to(self.device) 
                                        
                                        )
                

                    images_Anchor=val_ds.tensors[0] 

                    images_Positive=val_ds.tensors[1]
                    images_Negative=val_ds.tensors[2]
                
                 
                
                     
                
                

  
                      
                    
                                                                                                                         

                    losses_, nums_   =  \
                    self.loss_batch_00( dscrm_model,\
                           (images_Anchor ,images_Positive,images_Negative ),\
                            self._optimizer)      

                    losses.append(losses_)
                    nums.append(nums_ )
                print("]", end='')


                sum_nums = np.sum(nums)
                val_loss = np.sum(np.multiply(losses, nums)) / sum_nums
                    #acc = np.sum(np.multiply(accs, nums)) / sum_nums
                #################################################
                history.add_epoch_values(epoch, {'loss': loss, 'val_loss': val_loss})
                
                print(' - Loss: {:.6f}'.format(loss), end='')
                print(' - Test-loss: {:.6f}'.format(val_loss), end='')
            else:
                history.add_epoch_values(epoch, {'loss': loss })
                print(' - Loss: {:.6f}'.format(loss), end='')
                
            print("")
            
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history
    
###########################################
 