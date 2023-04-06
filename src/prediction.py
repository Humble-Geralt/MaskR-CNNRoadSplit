import torchvision
import utils
import torchvision.transforms as transforms
from lib.DataAPI import *
import cv2

#送入神经网络，获取预测结果
def GetPrediction(img,model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transf = transforms.ToTensor()
    img = transf(img)
    with torch.no_grad():
        prediction=model([img.to(device)])
    return prediction

#转换结果为Gray图像
def Maks2Gray(img):
    img_BGR=cv2.merge((img,img,img))
    img_GRAY=cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    return img_GRAY

def GetMask(prediction):
    mask=prediction[0]['masks']
    mask = mask[0][0].mul(255).byte().cpu().numpy()
    mask = Maks2Gray(mask)
    return mask

def GetResult(img,model_PATH = '../log/epoch-9-model.pkl'):
    #得到预测结果，返回一个tonser张量
    model = torch.load(model_PATH)
    prediction=GetPrediction(img,model)
    #获取所有mask
    maskAll=GetMask(prediction)

    return maskAll

if __name__ == '__main__':
    img = cv2.imread('../test/NewYork-2.png')
    mask = GetResult(img,'../log/ResNet50-ROI64-epoch-20-model.pkl')
    # cv2.namedWindow('final', cv2.WINDOW_NORMAL)

    cv2.imshow('res', img)
    cv2.imshow('mask_all', mask)

    ret2, th2 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('thotsu',th2)

    cv2.waitKey(0)
