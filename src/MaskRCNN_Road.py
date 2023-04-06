import torchvision
import utils
import lib.transforms as T
from lib.DataAPI import *
from lib.engine import train_one_epoch,evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def GetInstanceSegmentationModel(NumClasses):
    mask_roi_pool_cur = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=64,
                sampling_ratio=2)

    #加载在COCO数据集预训练好的backbone
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,mask_roi_pool = mask_roi_pool_cur)

    #获取模型的输入参数
    InFeatures = model.roi_heads.box_predictor.cls_score.in_features
    #替换已经训练好的模型头部

    model.roi_heads.box_predictor = FastRCNNPredictor(InFeatures,NumClasses)
    #获取输入mask分类器的输入参数数量
    InFeaturesMask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #定义隐层数量
    hidden_layer = 256
    #替换mask分类头
    model.roi_heads.mask_predictor =MaskRCNNPredictor(InFeaturesMask,hidden_layer,NumClasses)

    return model

def GetTransform(train):
    transforms = []
    #转换图像到张量
    transforms.append(T.ToTensor())
    if train:
        #训练过程中随即反转图像，进行数据增强
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    #加载数据集
    dataset = deepglobledataset('../dataset/deepglobe/train',GetTransform(train=True))
    datasetTest = deepglobledataset('../dataset/deepglobe/train',GetTransform(train=False))

    #切分数据集为训练集和测试集，后五十份数据为测试集
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset,indices[:-50])
    dataset_test = torch.utils.data.Subset(datasetTest,indices[-50:])


    #定义训练数据加载器和测试数据加载器
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,num_workers=4,collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,num_workers=4,collate_fn=utils.collate_fn)

    #使用CUDA加速以GPU进行训练，否则使用CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #设定检测类，本例只有道路和背景，所以为2
    NumClasses = 2

    #获取修改好的模型
    model = GetInstanceSegmentationModel(NumClasses)
    #加载至GPU
    model.to(device)

    #设置优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)

    #设定每3个epochs之后降低10倍学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    #训练
    num_epochs = 20
    for epoch in range(num_epochs):
        train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=10)

        #更新学习率
        lr_scheduler.step()

        #测试
        evaluate(model, data_loader_test, device=device)

          # 保存模型
        torch.save(model, '../log/ResNet50-ROI64-epoch-{}-model.pkl'.format(epoch+1))


    img, _ = dataset_test[0]
    model.eval()
    with torch.no_grad():
        prediction=model([img.to(device)])
    print(prediction)
    Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy()).show()
    Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()).show()
    print('Done!')

if __name__=="__main__":
    main()