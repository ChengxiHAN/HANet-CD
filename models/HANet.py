"""
HANet (Chengxi Han韩承熙,Wuhan University,chengxihan@whu.edu.cn,https://chengxihan.github.io/)
please Cited the paper:
C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“HANET: A HIERARCHICAL ATTENTION NETWORK FOR CHANGE DETECTION WITH BI-TEMPORAL VERY-HIGH-RESOLUTION REMOTE SENSING IMAGES,” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        # print('Channel为：',C)
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        # print('attention大小：', attention.shape)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        # print('矩阵大小1：',out.shape)
        out = out.view(m_batchsize, C, height, width)
        # print('矩阵大小2：', out.shape)
        out = self.gamma * out + x
        # print('矩阵大小3：', out.shape)
        return out


class Conv_CAM_Layer(nn.Module):

    def __init__(self, in_ch, out_in,use_pam=False):
        super(Conv_CAM_Layer, self).__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            CAM_Module(32),
            nn.Conv2d(32, out_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_in),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)


class FEC(nn.Module):
    """feature extraction cell"""
    #convolutional block
    def __init__(self, in_ch, mid_ch, out_ch):
        super(FEC, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1,bias=True)
        # self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=3, dilation=3, bias=True)  #加入dilation和padding！！
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # print('x的大小:',x.shape)
        x = self.conv1(x)
        # print('x的大小2:', x.shape)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


from torch.nn import Softmax
#实现轴向注意力中的 row Attention  // row_attn = RowAttention(in_dim = 8, q_k_dim = 4,device = device).to(device)
class RowAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim,use_pam=False):
    # def __init__(self, in_dim, q_k_dim, device):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        # self.device = device

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        # self.gamma = nn.Parameter(torch.zeros(1)).to(self.device)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
        # print('Q大小：', Q.shape)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w)
        # print('K大小：', K.shape)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w)
        # print('V大小：', V.shape)

        # size = (b*h,w,w) [:,i,j] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有h的第 Wj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        row_attn = torch.bmm(Q, K)
        # print('row_attn大小1：', row_attn.shape)
        ########
        # 此时的 row_atten的[:,i,0:w] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有行的 所有列(0:w)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:w）逐个位置的值的乘积，得到行attn
        ########

        # 对row_attn进行softmax
        row_attn = self.softmax(row_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，
        # print('row_attn大小2：', row_attn.shape)
        # size = (b*h,c1,w) 这里先需要对row_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 row_attn的行的乘积，即求权重和
        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        # print('out大小1：', out.shape)

        # size = (b,c1,h,2)
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
        # print('out大小2：', out.shape)

        out = self.gamma * out + x
        # print('out大小3：', out.shape)
        return out

#实现轴向注意力中的 column Attention  // col_attn = ColAttention(8, 4, device = device)
class ColAttention(nn.Module):

    # def __init__(self, in_dim, q_k_dim, device):
    def __init__(self, in_dim, q_k_dim,use_pam=False):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        # self.device = device

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        # self.gamma = nn.Parameter(torch.zeros(1)).to(self.device)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)

        # size = (b*w,h,h) [:,i,j] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的第 Hj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        col_attn = torch.bmm(Q, K)
        ########
        # 此时的 col_atten的[:,i,0:w] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的 所有列(0:h)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:h）逐个位置的值的乘积，得到列attn
        ########

        # 对row_attn进行softmax
        col_attn = self.softmax(col_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*w,c1,h) 这里先需要对col_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 col_attn的行的乘积，即求权重和
        out = torch.bmm(V, col_attn.permute(0, 2, 1))

        # size = (b,c1,h,w)
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)

        out = self.gamma * out + x

        return out


# class FEBlock1HCX(nn.Module):
class HAN(nn.Module):
    """HANet"""
    def __init__(self, in_ch=3, ou_ch=2):
        super(HAN, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 40  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 32, 64, 128, 256, 512

        self.conv0_0 = nn.Conv2d(3, n1, kernel_size=5, padding=2, stride=1)
        self.conv0 = FEC(filters[0], filters[0], filters[0])
        self.conv2 = FEC(filters[0], filters[1], filters[1])
        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.conv6 = nn.Conv2d(600, filters[1], kernel_size=1, stride=1)
        self.conv7 = nn.Conv2d(filters[1], ou_ch, kernel_size=3, padding=1, bias=False)

        self.conv6_1_1 = nn.Conv2d(filters[0] * 2, filters[0], padding=1, kernel_size=3, groups=filters[0] // 2,dilation=1)
        self.conv6_1_2 = nn.Conv2d(filters[0] * 2, filters[0], padding=2, kernel_size=3, groups=filters[0] // 2,dilation=2)
        self.conv6_1_3 = nn.Conv2d(filters[0] * 2, filters[0], padding=3, kernel_size=3, groups=filters[0] // 2,dilation=3)
        self.conv6_1_4 = nn.Conv2d(filters[0] * 2, filters[0], padding=4, kernel_size=3, groups=filters[0] // 2,dilation=4)
        self.conv1_1 = nn.Conv2d(filters[0] * 4, filters[0], kernel_size=1, stride=1)

        self.conv6_2_1 = nn.Conv2d(filters[1] * 2, filters[1], padding=1, kernel_size=3, groups=filters[1] // 2, dilation=1)
        self.conv6_2_2 = nn.Conv2d(filters[1] * 2, filters[1], padding=2, kernel_size=3, groups=filters[1] // 2, dilation=2)
        self.conv6_2_3 = nn.Conv2d(filters[1] * 2, filters[1], padding=3, kernel_size=3, groups=filters[1] // 2, dilation=3)
        self.conv6_2_4 = nn.Conv2d(filters[1] * 2, filters[1], padding=4, kernel_size=3, groups=filters[1] // 2, dilation=4)
        self.conv2_1 = nn.Conv2d(filters[1] * 4, filters[1], kernel_size=1, stride=1)

        self.conv6_3_1 = nn.Conv2d(filters[2] * 2, filters[2], padding=1, kernel_size=3, groups=filters[2] // 2, dilation=1)
        self.conv6_3_2 = nn.Conv2d(filters[2] * 2, filters[2], padding=2, kernel_size=3, groups=filters[2] // 2, dilation=2)
        self.conv6_3_3 = nn.Conv2d(filters[2] * 2, filters[2], padding=3, kernel_size=3, groups=filters[2] // 2, dilation=3)
        self.conv6_3_4 = nn.Conv2d(filters[2] * 2, filters[2], padding=4, kernel_size=3, groups=filters[2] // 2, dilation=4)
        self.conv3_1 = nn.Conv2d(filters[2] * 4, filters[2], kernel_size=1, stride=1)

        self.conv6_4_1 = nn.Conv2d(filters[3]*2, filters[3], padding=1, kernel_size=3, groups=filters[3]//2, dilation=1)
        self.conv6_4_2 = nn.Conv2d(filters[3]*2, filters[3], padding=2, kernel_size=3, groups=filters[3]//2, dilation=2)
        self.conv6_4_3 = nn.Conv2d(filters[3]*2, filters[3], padding=3, kernel_size=3, groups=filters[3]//2, dilation=3)
        self.conv6_4_4 = nn.Conv2d(filters[3]*2, filters[3], padding=4, kernel_size=3, groups=filters[3]//2, dilation=4)
        self.conv4_1 = nn.Conv2d(filters[3]*4, filters[3], kernel_size=1, stride=1)

        # SA
        self.cam_attention_1 = Conv_CAM_Layer(filters[0], filters[0], False)  #SA4
        self.cam_attention_2 = Conv_CAM_Layer(filters[1], filters[1], False)  #SA3
        self.cam_attention_3 = Conv_CAM_Layer(filters[2], filters[2], False)  #SA2
        self.cam_attention_4 = Conv_CAM_Layer(filters[3], filters[3], False)  #SA1

        #Row Attention
        self.row_attention_1 = RowAttention(filters[0], filters[0], False)  # SA4
        self.row_attention_2 = RowAttention(filters[1], filters[1], False)  # SA3
        self.row_attention_3 = RowAttention(filters[2], filters[2], False)  # SA2
        self.row_attention_4 = RowAttention(filters[3], filters[3], False)  # SA1

        # Col Attention
        self.col_attention_1 = ColAttention(filters[0], filters[0], False)  # SA4
        self.col_attention_2 = ColAttention(filters[1], filters[1], False)  # SA3
        self.col_attention_3 = ColAttention(filters[2], filters[2], False)  # SA2
        self.col_attention_4 = ColAttention(filters[3], filters[3], False)  # SA1

        self.c4_conv = nn.Conv2d(filters[3], filters[1], kernel_size=3, padding=1)
        self.c3_conv = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.c2_conv = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.c1_conv = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)

        self.pool1 = nn.AdaptiveAvgPool2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d(32)

        self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x1, x2=None):
    #     if x2 == None:
    #         x2 = x1
    def forward(self, x1, x2):

        # The first branch
        x1 = self.conv0(self.conv0_0(x1)) # Output of the first scale
        x3 = self.conv2(self.pool1(x1))
        x4 = self.conv4(self.pool2(x3))
        A_F4 = self.conv5(self.pool3(x4))

        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool1(x2))
        x6 = self.conv4(self.pool2(x5))
        A_F8 = self.conv5(self.pool3(x6))

        
        print('现在用的模块是：HANet-WHU-Fixed15-Epo50')
        c4_1 = self.conv4_1(
            torch.cat([self.conv6_4_1(torch.cat([A_F4, A_F8], 1)), self.conv6_4_2(torch.cat([A_F4, A_F8], 1)),
                       self.conv6_4_3(torch.cat([A_F4, A_F8], 1)), self.conv6_4_4(torch.cat([A_F4, A_F8], 1))], 1))
        c4 = self.cam_attention_4(c4_1) + self.row_attention_4(self.col_attention_4(c4_1))
        

        c3_1 = (self.conv3_1(torch.cat(
            [self.conv6_3_1(torch.cat([x4, x6], 1)), self.conv6_3_2(torch.cat([x4, x6], 1)),
             self.conv6_3_3(torch.cat([x4, x6], 1)), self.conv6_3_4(torch.cat([x4, x6], 1))], 1)))
        c3 = torch.cat([(self.cam_attention_3(c3_1)+self.row_attention_3(self.col_attention_3(c3_1))), self.Up1(c4)], 1)
        

        c2_1 = (self.conv2_1(torch.cat(
            [self.conv6_2_1(torch.cat([x3, x5], 1)), self.conv6_2_2(torch.cat([x3, x5], 1)),
             self.conv6_2_3(torch.cat([x3, x5], 1)), self.conv6_2_4(torch.cat([x3, x5], 1))], 1)))
        c2 = torch.cat([(self.cam_attention_2(c2_1)+self.row_attention_2(self.col_attention_2(c2_1))), self.Up1(c3)], 1)
        
        c1_1 = (self.conv1_1(torch.cat(
            [self.conv6_1_1(torch.cat([x1, x2], 1)), self.conv6_1_2(torch.cat([x1, x2], 1)),
             self.conv6_1_3(torch.cat([x1, x2], 1)), self.conv6_1_4(torch.cat([x1, x2], 1))], 1)))
        c1 = torch.cat([(self.cam_attention_1(c1_1)+self.row_attention_1(self.col_attention_1(c1_1))), self.Up1(c2)], 1)

        c1 = self.conv6(c1)
        out1 = self.conv7(c1)

        # feature_map = out1.squeeze(0)  # [1, 64, 112, 112] -> [64, 112, 112]
        #
        # feature_map_num = feature_map.shape[0]  # 返回通道数
        # for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出
        #     single_dim = feature_map[index]  # shape[256, 256]
        #     single_dim = single_dim.cpu().numpy()
        #     if index == 2:
        #         plt.imshow(single_dim, cmap='hot')
        #         # plt.imshow(single_dim, cmap='viridis')
        #         plt.axis('off')
        #         plt.show()

        return (out1,)



if __name__ == '__main__':
    print('hello sigma')
    x1 = torch.rand(5, 3, 256, 256)
    x2 = torch.rand(5, 3, 256, 256)
    model_restoration = FEBlock1HCX(3, 2)
    output = model_restoration(x1,x2)
    print(output[0].shape)

#测试模型大小
    # print('hello sigma')
    # input_size = 256
    # model_restoration = FEBlock1HCX(3, 2)
    # from ptflops import get_model_complexity_info
    # from torchstat import stat

    # # input = torch.rand((3, input_size, input_size))
    # # output = model_restoration(input)
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)

    # stat(model_restoration, (3, input_size, input_size))
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

