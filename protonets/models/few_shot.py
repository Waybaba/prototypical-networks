import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample):
        '''
        input: one set of data, including support sample and query sample
        return: loss, info{}
        '''
        xs = Variable(sample['xs']) # support 
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0) # concat support and query set [n_class*(n_support+n_query), ...]

        z = self.encoder.forward(x) # encoder all data points
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1) # centers [n_class, z_dim]
        zq = z[n_class*n_support:] # split for query data. [n_class*n_query, z_dim]

        dists = euclidean_dist(zq, z_proto) # [n_class*n_query, n_class]

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1) # [n_class, n_query, n_class]

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() # gather out the dim of label # loss = y*log(\hat y)

        _, y_hat = log_p_y.max(2) # cal y_hat for acc
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

class Protonet_2(nn.Module):
    def __init__(self, encoder_1, encoder_2):
        super(Protonet_2, self).__init__()
        
        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2

    def loss(self, sample):
        '''
        input: one set of data, including support sample and query sample
        return: loss, info{}
        '''
        xs = Variable(sample['xs']) # support 
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()
        '''
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0) # concat support and query set [n_class*(n_support+n_query), ...]

        z = self.encoder.forward(x) # encoder all data points
        This is raw version, 
        '''
        # here, we use different encoder for support and query set
        # why do we concat them up? as the raw version use the whole z 
        # for post process, so we concat them to stay consistent.
        z_1 = self.encoder_1(xs.view(n_class * n_support, *xs.size()[2:]))
        z_2 = self.encoder_2(xq.view(n_class * n_query, *xq.size()[2:]))
        z = torch.cat([z_1, z_2], dim=0)
        

        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1) # centers [n_class, z_dim]
        zq = z[n_class*n_support:] # split for query data. [n_class*n_query, z_dim]

        dists = euclidean_dist(zq, z_proto) # [n_class*n_query, n_class]

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1) # [n_class, n_query, n_class]

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() # gather out the dim of label # loss = y*log(\hat y)

        _, y_hat = log_p_y.max(2) # cal y_hat for acc
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)

@register_model('protonet_conv_2')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder_1 = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    encoder_2 = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet_2(encoder_1, encoder_2)