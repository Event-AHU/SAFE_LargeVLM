# SAFE_LargeVLM

**emantic-Aware Frame-Event Fusion based Pattern Recognition via Large Vision-Language Models**, Xiao Wang, Dong Li, Jiandong Jin, Yuhao Zhang, Yanlin Zhong, Yaoyang Wu, Lan Chen. [[arXiv](https://arxiv.org/abs/2308.11937)] [[Paper_with_Code](https://paperswithcode.com/paper/learning-bottleneck-transformer-for-event#code)] [[Poster](https://github.com/Event-AHU/EFV_event_classification/blob/main/Poster_EFV_event_classification.pdf)] 


## :dart: Abstract 
Pattern recognition through the fusion of RGB frames and Event streams has emerged as a novel research area in recent years. Current methods typically employ backbone networks to individually extract the features of RGB frames and event streams, and subsequently fuse these features for pattern recognition. However, we posit that these methods may suffer from two key issues: 1). They attempt to directly learn a mapping from input vision modalities to the semantic labels. This approach often leads to sub-optimal results due to the disparity between the input and semantic labels; 2). They utilize small-scale backbone networks for the extraction of RGB and Event input features, thus these models fail to harness the recent performance advancements of large-scale visual-language models. In this study, we introduce a novel pattern recognition framework that consolidates the semantic labels, RGB frames, and event streams, leveraging pre-trained large-scale visionlanguage models. Specifically, given the input RGB frames, event streams, and all the predefined semantic labels, we employ a pre-trained large-scale vision model (CLIP vision encoder) to extract the RGB and event features. To handle the semantic labels, we initially convert them into language descriptions through prompt engineering, and then obtain the semantic features using the pre-trained large-scale language model (CLIP text encoder). Subsequently, we integrate the RGB/Event features and semantic features using multimodal Transformer networks. The resulting frame and event tokens are further amplified using self-attention layers. Concurrently, we propose to enhance the interactions between text tokens and RGB/Event to kens via cross-attention. Finally, we consolidate all three modalities using self-attention and feed-forward layers for recognition. Comprehensive experiments on the HARDVS and PokerEvent datasets fully substantiate the efficacy of our proposed SAFE model.


<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/main/figure/firstIMG.jpg" alt="feature_vis"/>
</p> 


## :construction_worker: Environment Setting 
```   
Python 3.9
torch  2.0.1
easydict 1.10
ftfy   6.1.1
Jinja2 3.1.2
scipy  1.11.2
tqdm   4.66.1
numpy  1.26.0
Pillow 10.0.1
torchvision 0.15.2
sentence-transformers  2.2.2
```

## :triangular_flag_on_post: Framework 
<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/main/figure/frameworkV4.jpg" alt="feature_vis"/>
</p> 

## :floppy_disk: Dataset Download and Pre-processing 
### HARDVS
```
[Event Images] 链接：https://pan.baidu.com/s/1OhlhOBHY91W2SwE6oWjDwA?pwd=1234 提取码：1234

[Compact Event file] 链接：https://pan.baidu.com/s/1iw214Aj5ugN-arhuxjmfOw?pwd=1234 提取码：1234
```

### POKER
```
BaiduYun (178GB): 链接：https://pan.baidu.com/s/1vQnHZUqQ1o58SajvtE-uHw?pwd=AHUE 提取码：AHUE 

DropBox (178GB): https://www.dropbox.com/scl/fo/w658kwhfi3qa8naul3eeb/h?rlkey=zjn4b69wa1e3mhid8p6hh8v75&dl=0
```

## :hourglass: Training and Testing 
```
    train
    CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=30000 train.py poker

    test
    CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=30000 eval.py poker
```

## :chart_with_upwards_trend: Experimental Results 
<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/main/figure/ASLtop5TSNE.jpg" alt="feature_vis"/>
</p> 
## :newspaper:License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## :newspaper:Acknowledgement

Our code is implemented based on VTB,VTF

## :newspaper: Citation 

If you think this work contributes to your research, please give us a star :star2: and cite this paper via: 
```
@
```

If you have any questions about this work, please leave an issue. 










