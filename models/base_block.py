import torch.nn as nn
import torch
import torch.nn.functional as F
from CLIP.clip import clip
from models.vit import *
from CLIP.CoOp import *
import pdb 
device = "cuda" if torch.cuda.is_available() else "cpu"

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 计算查询、键和值
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.input_dim).float())  # 缩放注意力分数
        
        # 应用 softmax 函数获取注意力权重
        attention_weights = self.softmax(attention_scores)
        
        # 使用注意力权重对值进行加权求和
        attended_values = torch.matmul(attention_weights, V)
        
        return attended_values
    

class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim,in_q_dim,hid_q_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_q_dim = in_q_dim #新增
        self.hid_q_dim = hid_q_dim #新增
        # 定义查询、键、值三个线性变换
        self.query = nn.Linear(in_q_dim, hid_q_dim, bias=False) #变化
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, x, y):
        # 对输入进行维度变换，为了方便后面计算注意力分数
        batch_size = x.shape[0]   # batch size
        num_queries = x.shape[1]  # 查询矩阵中的元素个数
        num_keys = y.shape[1]     # 键值矩阵中的元素个数
        x = self.query(x)  # 查询矩阵
        y = self.key(y)    # 键值矩阵
        # 计算注意力分数
        attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (self.out_dim ** 0.5)  # 计算注意力分数，注意力分数矩阵的大小为 batch_size x num_queries x num_keys x num_keys
        attn_weights = F.softmax(attn_scores, dim=-1)  # 对注意力分数进行 softmax 归一化
        # 计算加权和
        V = self.value(y)  # 通过值变换得到值矩阵 V
        output = torch.bmm(attn_weights, V)  # 计算加权和，output 的大小为 batch_size x num_queries x num_keys x out_dim
       
        return output


class TransformerClassifier(nn.Module):
    def __init__(self, attr_num,attr_words, dim=768, pretrain_path='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTB-main/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(512, dim)
        self.visual_embed= nn.Linear(512, dim)#1
        self.vit = vit_base()
        self.vit.load_param(pretrain_path)
        self.blocks = self.vit.blocks[-1:]
        self.norm = self.vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.remain_weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(197)])
        self.bn = nn.BatchNorm1d(self.attr_num)
        #self.text = clip.tokenize(attr_words,prompt ="A photo of a ").to(device)#1
        self.text = clip.tokenize(attr_words).to(device)
        self.rgb_embed   = nn.Parameter(torch.zeros(1, 1, dim))
        self.event_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.self_attention = SelfAttention(768) 
        # 实例化CrossAttention对象
        self.cross_model = CrossAttention(in_dim=768, out_dim=768, in_q_dim=768, hid_q_dim=768)#除了第一个维度，其他自定
        # self.cross_model.to(device)
        
        self.CLS_token = nn.Parameter(torch.zeros(1, 10, dim))
        self.fc1 = nn.Linear(768*10, self.attr_num)
        

    def forward(self, rgb_videos, event_videos, ViT_model):
        rgb_ViT_features=[]
        event_ViT_features=[]
        if len(rgb_videos.size())<5 :
            rgb_videos.unsqueeze(1) 
        
        batch_size, num_frames, channels, height, width = rgb_videos.size() 
        rgb_imgs = rgb_videos.view(-1, channels, height, width) 
        event_imgs = event_videos.view(-1, channels, height, width)

        #CLIP 提取视频帧特征
        
        for img in rgb_imgs:
            img = img.unsqueeze(0)
            img = img.to(device)
            rgb_ViT_features.append(ViT_model.encode_image(img).squeeze(0))
            #消融2（大模型换常规模型）
            #rgb_ViT_features.append(self.vit(img).squeeze(0))
            #breakpoint() 
        rgb_ViT_image_features = torch.stack(rgb_ViT_features).to(device).float()#消融是[5, 197, 768],但clip是[5, 197, 512]
        
        for img in event_imgs:
            img = img.unsqueeze(0)
            img = img.to(device)
            event_ViT_features.append(ViT_model.encode_image(img).squeeze(0))
            #消融2（大模型换常规模型）
            #event_ViT_features.append(self.vit(img).squeeze(0)) 
        event_ViT_image_features = torch.stack(event_ViT_features).to(device).float()
        #breakpoint()
        _, token_num, visual_dim = rgb_ViT_image_features.size()
        rgb_ViT_image_features   = rgb_ViT_image_features.view(batch_size, num_frames, token_num, visual_dim) 
        event_ViT_image_features = event_ViT_image_features.view(batch_size, num_frames, token_num, visual_dim)
        
        rgb_ViT_image_features = self.visual_embed(torch.mean(rgb_ViT_image_features, dim=1)) 
        event_ViT_image_features = self.visual_embed(torch.mean(event_ViT_image_features, dim=1))

        text_features = ViT_model.encode_text(self.text).to(device).float()#这里直接不concat text做消融
        textual_features = self.word_embed(text_features).expand(rgb_ViT_image_features.shape[0], text_features.shape[0], 768)  
        
        tex_embed   = textual_features + self.tex_embed               #torch.Size([2, 300, 768])
        rgb_embed   = rgb_ViT_image_features + self.rgb_embed       #torch.Size([2, 197, 768])
        event_embed = event_ViT_image_features + self.event_embed

        x = torch.cat([tex_embed, rgb_embed], dim=1) #torch.Size([2, 497, 768])
        x = torch.cat([x, event_embed], dim=1) 

        #消融3，多模型融合不做，直接用concat，注释下面三行(Multi-modal Transformer)
        for b_c, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x) #torch.Size([32, 694, 768])
        
        texttoken_features      = x[:, :self.attr_num, :]     #0到299    ([1, 114, 768])
        rgb_imgtoken_features   = x[:, self.attr_num:self.attr_num+rgb_embed.shape[1], :]    #300+197 ([16, 197, 768])
        event_imgtoken_features = x[:, self.attr_num+rgb_embed.shape[1]:, :] #498~694
        
        ############################################################
        ## self-attention and cross-attention module 
        ############################################################

        vis_event_tokens = torch.cat((rgb_imgtoken_features, event_imgtoken_features), dim=1)#([1, 394, 768])
        #消融4(self attention)
        vis_event_tokens = self.self_attention(vis_event_tokens)#([1, 394, 768])

        enhanced_vis_tokens   = vis_event_tokens[:, :197, :] #[1, 197, 768]
        enhanced_event_tokens = vis_event_tokens[:, 197:, :] #[1, 197, 768]
        enhanced_features = enhanced_vis_tokens + enhanced_event_tokens 

        #消融1(Semantic Attribute Set)
        #texttoken_features = torch.zeros_like(texttoken_features)
        
        cross_output1 = self.cross_model(texttoken_features, enhanced_vis_tokens)  #[1, 114, 768]
        cross_output2 = self.cross_model(texttoken_features, enhanced_event_tokens)#[1, 114, 768]
        cross_features = cross_output1 + cross_output2
                        
        visualTokens = torch.cat((enhanced_features, cross_features), dim=1) #[1, 311, 768]

        #消融5(cross attention)
        #visualTokens = torch.cat((enhanced_features, texttoken_features), dim=1) #[1, 311, 768]
        
        # 获取visualTokens的实际数量
        actual_num = visualTokens.size(0)

        # 扩展self.CLS_token的第一维为实际数量
        expanded_CLS_token = self.CLS_token[:actual_num].expand(actual_num, -1, -1)
        #breakpoint()
        visualTokens = torch.cat((visualTokens, expanded_CLS_token), dim=1)  #([1, 321, 768])
                
        final_features =  self.self_attention(visualTokens) #([1, 321, 768])

        #消融4
        #final_features =  visualTokens 
        l = final_features.shape[1]
        final_features_CLS = final_features[:, l-10:, :]
        
        output_feat = final_features_CLS.view(final_features_CLS.shape[0], -1)

        # 全连接层
        output_feat = self.fc1(output_feat)
        
        # softmax归一化
        logits = F.log_softmax(output_feat, dim=1) 
        #breakpoint()
        return logits

