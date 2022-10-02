'''
Author: ----
Date: 2022-04-08 11:09:07
LastEditors: GhMa
LastEditTime: 2022-10-02 19:26:00
'''
import torch
import torch.nn as nn
from .snn_modules import tdLayer, BN, Seq2ANN


def reset_snn(net: nn.Module):
    for m in net.modules():
        if hasattr(m, '_reset'):
            m._reset()


# blocks ################################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, 
        inplanes, planes, 
        stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer='bn', 
        spiking_neuron: callable = None, **kwargs
    ):
        super(BasicBlock, self).__init__()
        if norm_layer == 'bn':
            norm_layer = BN
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sn = spiking_neuron(**kwargs)

        if norm_layer is not None:
            self.conv1 = tdLayer(
                conv3x3(inplanes, planes, stride), 
                norm_layer((planes))
            )
            
            self.conv2 = tdLayer(
                conv3x3(planes, planes),
                norm_layer(planes)
            )
        else:
            self.conv1 = tdLayer(conv3x3(inplanes, planes, stride), )
            self.conv2 = tdLayer(conv3x3(planes, planes), )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.sn(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.sn(out)
        return out


# Networks ################################################################
class SpikingResNet(nn.Module):
    r"""
    128 width S-ResNet
    """
    def __init__(
        self, 
        block, layers, 
        num_classes=1000, 
        zero_init_residual=False,
        groups=1, 
        width_per_group=64, 
        replace_stride_with_dilation=None,
        norm_layer='bn', 
        spiking_neuron: callable = None, 
        n_input = [3, 32, 32],
        **kwargs
    ):
        super(SpikingResNet, self).__init__()
        if norm_layer == 'bn':
            norm_layer = BN
        self._norm_layer = norm_layer
        self.in_dim = n_input
        self.inplanes = 64
        self.dilation = 1
        self.T = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.sn = spiking_neuron(**kwargs)

        if norm_layer is not None:
            self.conv1 = tdLayer(
                nn.Conv2d(
                    n_input[0], self.inplanes, 
                    kernel_size=3, stride=1, padding=1, bias=False
                ),
                norm_layer(self.inplanes)
            )
        else:
            self.conv1 = tdLayer(
                nn.Conv2d(
                    n_input[0], self.inplanes, 
                    kernel_size=3, stride=1, padding=1, bias=False
                ),
            )
        
        self.layer1 = self._make_layer(
            block, 128, layers[0], 
            spiking_neuron=spiking_neuron, 
            **kwargs
        )
        self.layer2 = self._make_layer(
            block, 256, layers[1], 
            stride=2,
            dilate=replace_stride_with_dilation[0], 
            spiking_neuron=spiking_neuron, 
            **kwargs
        )
        self.layer3 = self._make_layer(
            block, 512, layers[2], 
            stride=2,
            dilate=replace_stride_with_dilation[1], 
            spiking_neuron=spiking_neuron, 
            **kwargs
        )
        self.avgpool = tdLayer(
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc1 = tdLayer(
            nn.Linear(512 * block.expansion, 256)
        )
        self.fc2 = tdLayer(
            nn.Linear(256, num_classes)
        )
        
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu'
                    )

    def _make_layer(
        self, 
        block, planes, blocks, 
        stride=1, dilate=False, 
        spiking_neuron: callable = None, 
        **kwargs
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if norm_layer is not None:
                downsample = tdLayer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion)
                )
            else:
                downsample = tdLayer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer, 
                spiking_neuron, 
                **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation,
                    norm_layer=norm_layer, 
                    spiking_neuron=spiking_neuron, 
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.sn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        x = self.sn(x)
        x = self.fc2(x)
        # mod 0408 15.24
        # x = self.sn(x, out_u=True)
        return x

    def forward(self, x):
        """ if x.dim() == 2:
            # [N, C*H*W]
            bsz = x.size(0)
            x = x.view(bsz, self.in_dim[0], self.in_dim[1], self.in_dim[2]) """    
        return self._forward_impl(x)


class SpikingResNet_type2(nn.Module):
    r"""
    64-width S-ResNet
    """
    def __init__(
        self, 
        block, layers, 
        num_classes=1000, 
        zero_init_residual=False,
        groups=1, 
        width_per_group=64, 
        replace_stride_with_dilation=None,
        norm_layer='bn', 
        spiking_neuron: callable = None, 
        n_input = [3, 32, 32],
        **kwargs
    ):
        super(SpikingResNet_type2, self).__init__()
        if norm_layer == 'bn':
            norm_layer = BN
        self._norm_layer = norm_layer
        self.in_dim = n_input
        self.inplanes = 64
        self.dilation = 1
        self.T = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.sn = spiking_neuron(**kwargs)

        if n_input[1] == 32:
            conv1_ks = 3
            conv1_stride = 1
            conv1_padding = 1
        else:
            conv1_ks = 7
            conv1_stride = 2
            conv1_padding = 3

        if norm_layer is not None:    
            self.conv1 = tdLayer(
                nn.Conv2d(
                    n_input[0], self.inplanes, 
                    kernel_size=conv1_ks, 
                    stride=conv1_stride, 
                    padding=conv1_padding, 
                    bias=False
                ),
                norm_layer(self.inplanes)
            )
        else:
            self.conv1 = tdLayer(
                nn.Conv2d(
                    n_input[0], self.inplanes, 
                    kernel_size=conv1_ks, 
                    stride=conv1_stride,
                    padding=conv1_padding, 
                    bias=False
                ),
            )

        self.maxpool = tdLayer(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(
            block, 64, layers[0], 
            spiking_neuron=spiking_neuron, 
            **kwargs
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], 
            stride=2,
            dilate=replace_stride_with_dilation[0], 
            spiking_neuron=spiking_neuron, 
            **kwargs
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], 
            stride=2,
            dilate=replace_stride_with_dilation[1], 
            spiking_neuron=spiking_neuron, 
            **kwargs
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], 
            stride=2,
            dilate=replace_stride_with_dilation[1], 
            spiking_neuron=spiking_neuron, 
            **kwargs
        )

        self.avgpool = tdLayer(
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc1 = tdLayer(
            nn.Linear(512 * block.expansion, num_classes)
        )
        
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu'
                    )

    def _make_layer(
        self, 
        block, planes, blocks, 
        stride=1, dilate=False, 
        spiking_neuron: callable = None, 
        **kwargs
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if norm_layer is not None:
                downsample = tdLayer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion)
                )
            else:
                downsample = tdLayer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                ) 

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer, 
                spiking_neuron, 
                **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation,
                    norm_layer=norm_layer, 
                    spiking_neuron=spiking_neuron, 
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.sn(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _spiking_resnet(
    block, layers, pretrained, progress, 
    spiking_neuron, 
    n_input, n_output,
    **kwargs
):
    model = SpikingResNet(
        block, layers, 
        spiking_neuron=spiking_neuron, 
        n_input=n_input, 
        num_classes=n_output,
        **kwargs
    )
    
    return model


def _spiking_resnet_type2(
    block, layers, pretrained, progress, 
    spiking_neuron, 
    n_input, n_output,
    **kwargs
):
    model = SpikingResNet_type2(
        block, layers, 
        spiking_neuron=spiking_neuron, 
        n_input=n_input, 
        num_classes=n_output,
        **kwargs
    )
    
    return model


def spiking_resnet19(
    pretrained=False, 
    progress=True, 
    spiking_neuron: callable = None, 
    n_input = [3, 32, 32],
    n_output = 1000,
    norm_layer = 'bn',
    **kwargs
):
    r"""
    A spiking version of ResNet-19 model
    """

    return _spiking_resnet( 
        BasicBlock, 
        [3, 3, 2], 
        pretrained, 
        progress, 
        spiking_neuron,
        n_input,
        n_output,
        width_per_group=64,
        norm_layer=norm_layer,
        **kwargs
    )


def spiking_resnet18(
    pretrained=False, 
    progress=True, 
    spiking_neuron: callable = None, 
    n_input = [3, 32, 32],
    n_output = 1000,
    norm_layer = 'bn',
    **kwargs
):
    r"""
    A spiking version of ResNet-18 model
    """

    return _spiking_resnet_type2( 
        BasicBlock, 
        [2, 2, 2, 2], 
        pretrained, 
        progress, 
        spiking_neuron,
        n_input,
        n_output,
        width_per_group=64,
        norm_layer=norm_layer,
        **kwargs
    )


def spiking_resnet34(
    pretrained=False, 
    progress=True, 
    spiking_neuron: callable = None, 
    n_input = [3, 224, 224],
    n_output = 1000,
    norm_layer = 'bn',
    **kwargs
):
    r"""
    A spiking version of ResNet-34 model
    """

    return _spiking_resnet_type2( 
        BasicBlock, 
        [3, 4, 6, 3], 
        pretrained, 
        progress, 
        spiking_neuron,
        n_input,
        n_output,
        width_per_group=64,
        norm_layer=norm_layer,
        **kwargs
    )


class BasicConvLayer(nn.Module):
    r"""
    Building block for S-VGG
    Forked from https://github.com/Gus-Lab/temporal_efficient_training
    """
    def __init__(
        self,
        in_plane, out_plane,
        kernel_size, stride, padding,
        spiking_neuron=None,
        norm_layer=True,
        **kwargs
    ):
        super(BasicConvLayer, self).__init__()
        if norm_layer:
            self.fwd = Seq2ANN(
                nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
                nn.BatchNorm2d(out_plane)
            )
        else:
            self.fwd = Seq2ANN(
                nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            )
        self.sn = spiking_neuron(**kwargs)

    def forward(self, x):
        x = self.fwd(x)
        x = self.sn(x)
        return x

        
class VGGSNN(nn.Module):
    r"""
    Spiking VGG Net
    Forked from https://github.com/Gus-Lab/temporal_efficient_training
    """
    def __init__(
        self,
        n_input=[2, 48, 48],
        n_output=10,
        spiking_neuron=None,
        **kwargs
    ):
        super(VGGSNN, self).__init__()
        pool = Seq2ANN(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            BasicConvLayer(n_input[0], 64, 3, 1, 1, spiking_neuron, **kwargs),
            BasicConvLayer(64, 128, 3, 1, 1, spiking_neuron, **kwargs),
            pool,
            BasicConvLayer(128, 256, 3, 1, 1, spiking_neuron, **kwargs),
            BasicConvLayer(256, 256, 3, 1, 1, spiking_neuron, **kwargs),
            pool,
            BasicConvLayer(256, 512, 3, 1, 1, spiking_neuron, **kwargs),
            BasicConvLayer(512, 512, 3, 1, 1, spiking_neuron, **kwargs),
            pool,
            BasicConvLayer(512, 512, 3, 1, 1, spiking_neuron, **kwargs),
            BasicConvLayer(512, 512, 3, 1, 1, spiking_neuron, **kwargs),
            pool,
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier = Seq2ANN(nn.Linear(512*W*W, n_output))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGSNNwoAP(nn.Module):
    r"""
    Spiking VGG Net w/o AvgPooling
    Forked from https://github.com/Gus-Lab/temporal_efficient_training
    """
    def __init__(
        self,
        n_input=[2, 48, 48],
        n_output=10,
        spiking_neuron=None,
        **kwargs
    ):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            BasicConvLayer(n_input[0], 64, 3, 1, 1, spiking_neuron, **kwargs),
            BasicConvLayer(64, 128, 3, 2, 1, spiking_neuron, **kwargs),
            BasicConvLayer(128, 256, 3, 1, 1, spiking_neuron, **kwargs),
            BasicConvLayer(256, 256, 3, 2, 1, spiking_neuron, **kwargs),
            BasicConvLayer(256, 512, 3, 1, 1, spiking_neuron, **kwargs),
            BasicConvLayer(512, 512, 3, 2, 1, spiking_neuron, **kwargs),
            BasicConvLayer(512, 512, 3, 1, 1, spiking_neuron, **kwargs),
            BasicConvLayer(512, 512, 3, 2, 1, spiking_neuron, **kwargs),
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier = Seq2ANN(nn.Linear(512*W*W, n_output))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    
class CIFARNet(nn.Module):
    r"""
    CIFAR Net
    """
    def __init__(
        self,
        n_input=[3, 32, 32],
        n_output=10,
        spiking_neuron=None,
        **kwargs
    ):
        super(CIFARNet, self).__init__()
        pool = Seq2ANN(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            BasicConvLayer(n_input[0], 128, 3, 1, 1, spiking_neuron, False, **kwargs),
            BasicConvLayer(128, 256, 3, 1, 1, spiking_neuron, False, **kwargs),
            pool,
            BasicConvLayer(256, 512, 3, 1, 1, spiking_neuron, False, **kwargs),
            pool,
            BasicConvLayer(512, 1024, 3, 1, 1, spiking_neuron, False, **kwargs),
            BasicConvLayer(1024, 512, 3, 1, 1, spiking_neuron, False, **kwargs),
        )
        W = int(32 / 2 / 2)
        self.classifier = nn.Sequential(
            Seq2ANN(nn.Linear(512*W*W, 1024)),
            Seq2ANN(nn.Linear(1024, 512)),
            Seq2ANN(nn.Linear(512, n_output)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

