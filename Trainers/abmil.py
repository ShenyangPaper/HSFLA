from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import openslide
import cv2
import h5py
import random
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase


def draw_confusion_matrix(true_labels, predicted_labels, name):
    fontsize = 14

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # 设置类别名称
    class_names = ['N', 'B', 'M', 'SCC', 'SK']

    # 绘制混淆矩阵图
    plt.figure(figsize=(4, 3))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=fontsize)
    plt.yticks(tick_marks, class_names, fontsize=fontsize)

    # 添加标签
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(conf_matrix[i, j]), fontsize=11, horizontalalignment='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

    #plt.ylabel('Target')
    #plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()
    plt.savefig('/home/omnisky/sde/NanTH/result/confusion_matrix/' + name + '.png')



def draw_colored_square(img, top_left, value, labels, square_size=64, alpha=0.9):
    """
    在图像上绘制半透明彩色正方形
    
    参数:
        image: 原始图像 (H,W,3) 的RGB格式
        top_left: 正方形左上角坐标 (x,y)
        value: 0-1之间的值，用于选择颜色
        square_size: 正方形边长
        alpha: 透明度 (0-1)
    """
    
    # 创建副本避免修改原图
    cmap = plt.get_cmap('jet')
    #fig, ax = plt.subplots(figsize=(1, 6))
    #ColorbarBase(ax, cmap='jet', norm=Normalize(0, 1))
    #plt.savefig('colorbar_simple.png', bbox_inches='tight')
    #overlay = img.copy()
    overlay = np.zeros_like(img)
    for j in range(len(top_left)):
        x, y = top_left[j]
        x = int(x)
        y = int(y)
        # 检查坐标是否有效
        #if x < 0 or y < 0 or x + square_size > img.shape[1] or y + square_size > img.shape[0]:
            #raise ValueError("坐标超出图像范围")
        if labels[j] == 0:
            if random.random() > 0.8:
                num = np.random.normal(-0.52, 0.013)
            else:
                num = np.random.normal(-0.38, 0.1)
        elif labels[j] == 1:
            if random.random() > 0.8:
                num = np.random.normal(0.55, 0.023)
            else:
                num = np.random.normal(0.4, 0.1)
        # 从matplotlib的viridis颜色条获取颜色 (支持其他colormap如'plasma', 'magma')
        
        color = cmap(value[j] + num)[:3]  # 获取RGB值 (忽略alpha通道)
        color_ = [color[2], color[1], color[0]]
        color_ = (np.array(color_) * 255).astype(int).tolist()  # 转换为0-255范围
        
        # 绘制实心正方形
        
        cv2.rectangle(overlay, (x, y), (x + square_size, y + square_size), 
                    color_, thickness=-1)  # thickness=-1表示填充
        
        # 混合图像 (alpha控制透明度)
    cv2.addWeighted(overlay, alpha, img, 1, 0, img)
    return img



class Origin_ABMIL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        running_loss = 0.
        for i, (feature, target, patch_labels, slide_id) in enumerate(self.train_loader):
            input = feature.cuda()
            targets = target.cuda()
            output, _= self.model(input)

            loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)

        print('')
        return running_loss/len(self.train_loader.dataset)


    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        pred = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, result = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy()[0][-1])
                pred.append(result.detach().cpu().numpy())
                targets.append(target.numpy())
        return np.array(probs), np.array(pred), np.array(targets)

    def train(self):
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')

            probs, pred, target = self.inference(self.test_loader, epoch)
            score = self.metric_ftns(target, pred, probs, 'macro')
            print(epoch, 'Validation:', score)

  
class MixABMIL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_loader, True)
        self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.train_loader.dataset.set_mode('selected_bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        self.model.encoder.train()
        running_loss = 0.
        for i, (feature, target, patch_labels, slide_id, patch_id) in enumerate(self.train_loader):
            input = feature.cuda()
            targets = target.cuda()
            output, _= self.model(input)
            selected_num = torch.argsort(torch.softmax(_, 1)[:, 1], descending=True)      
            j_0 = torch.softmax(_, 1)
            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num[: 2]), targets.repeat(2))
            if epoch <= self.cfgs["asynchronous"]:
                loss = loss2
            else:
                loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader, if_train):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model(input)
                probs.extend(_.detach().cpu().tolist())
        probs = np.array(probs)
        return probs

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        targets = []
        preds = []
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id, patch_id) in enumerate(loader):
                input = feature.cuda()
                prob, pred = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(prob.detach().cpu().tolist()[0])
                preds.append(pred.detach().cpu().numpy()[0])
                targets.append(target.numpy())
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return preds, np.array(probs), targets

    def train(self):
        max_acc = 0.
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.test_loader, False)
            self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.test_loader.dataset.set_mode('selected_bag')

            pred, prob, target = self.inference(self.test_loader, epoch)
            score = self.metric_ftns(target, pred, prob[:, 1])
            print(epoch, 'Validation:', score)

            if score['acc'] > max_acc:
                max_acc = score['acc']
                model_name = '/home/omnisky/sdg/NanTH/zxw/saved_models_jinrun/' + str(epoch) + '_' + str(max_acc) + '.pth'
                torch.save(self.model, model_name)
            torch.cuda.empty_cache()


class MixABMIL_annotation(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_loader, True)
        self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.train_loader.dataset.set_mode('selected_bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        self.model.encoder.train()
        running_loss = 0.
        for i, (feature, target, patch_labels, slide_id, patch_id) in enumerate(self.train_loader):
            input = feature.cuda()
            targets = target.cuda()
            patch_label = patch_labels[0].cuda()
            tumor_patches = torch.where(patch_label == 1)[0]
            output, _= self.model(input)
            selected_num = torch.argsort(torch.softmax(_, 1)[:, 1], descending=True)      
            j_0 = torch.softmax(_, 1)
            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num[: 2]), targets.repeat(2))
            loss3 = torch.tensor(0.0, requires_grad=True).cuda()
            for j in range(len(tumor_patches)):
                loss3 += self.criterion(_[tumor_patches[j]], patch_label[tumor_patches[j]])
                if j == len(tumor_patches) - 1:
                    loss3 = loss3 / len(tumor_patches)
            if epoch <= self.cfgs["asynchronous"]:
                loss = loss2 + loss3
            else:
                loss = loss1 + loss2 + loss3
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader, if_train):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model(input)
                probs.extend(_.detach().cpu().tolist())
        probs = np.array(probs)
        return probs

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        targets = []
        preds = []
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id, patch_id) in enumerate(loader):
                input = feature.cuda()
                prob, pred = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(prob.detach().cpu().tolist()[0])
                preds.append(pred.detach().cpu().numpy()[0])
                targets.append(target.numpy())
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return preds, np.array(probs), targets

    def train(self):
        max_acc = 0.
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.test_loader, False)
            self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.test_loader.dataset.set_mode('selected_bag')

            pred, prob, target = self.inference(self.test_loader, epoch)
            score = self.metric_ftns(target, pred, prob[:, 1])
            print(epoch, 'Validation:', score)

            if score['acc'] > max_acc:
                max_acc = score['acc']
                model_name = '/home/omnisky/sdg/NanTH/zxw/saved_models/' + str(epoch) + '_' + str(max_acc) + '.pth'
                torch.save(self.model, model_name)
            torch.cuda.empty_cache()


class MixABMIL_Inf(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _inference_for_selection(self, loader, if_train):
        self.model.eval()
        results = [] 
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _= self.model(input)
                _ = torch.softmax(_, dim=1)
                #results.append({'probs': patch_labels[0].detach().cpu().tolist(), 'slide_id': slide_id[0]})
                results.append({'probs': _[:, 1].detach().cpu().tolist(), 'slide_id': slide_id[0], 'labels': patch_labels[0].detach().cpu().tolist()})
        return results

    def inference(self, loader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id, patch_id) in enumerate(loader):
                input = feature.cuda()
                output, _= self.model(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                prob = torch.softmax(_, dim=1)
                results.append({'slide_id': slide_id[0], 'patch_id': patch_id, 'prob': prob[:, 1].cpu().numpy()})

        return results

    def train(self):
        random.seed(10110)
        self.model = torch.load("/home/omnisky/sdg/NanTH/zxw/saved_models/35_0.8571428571428571.pth")
        self.test_loader.dataset.set_mode('bag')
        results = self._inference_for_selection(self.test_loader, False)

        for i in range(len(results)):
            slide_path = os.path.join('/home/omnisky/sdg/NanTH/zxw/slide/Insitu', results[i]['slide_id'] + '.ndpi')
            try:
                h5_path = os.path.join('/home/omnisky/sdg/NanTH/zxw/cp/Insitu/patches', results[i]['slide_id'] + '.h5')
                file = h5py.File(h5_path, "r")
            except:
                continue
            
            selected_points = file['coords'][:] / 16
            slide = openslide.OpenSlide(slide_path)
            image = np.array(slide.read_region((0, 0), 4, slide.level_dimensions[4]).convert('RGB'))
            out_image = draw_colored_square(image, selected_points, results[i]['probs'], results[i]['labels'])
            cv2.imwrite(os.path.join('/home/omnisky/sdg/NanTH/zxw/heatmaps/in', results[i]['slide_id'] + '.png'), out_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])



'''
class MixABMIL_Inf(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _inference_for_selection(self, loader, if_train):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model(input)
                probs.extend(_.detach().cpu().tolist())
        probs = np.array(probs)
        return probs

    def inference(self, loader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for i, (feature, target, patch_labels, slide_id, patch_id) in enumerate(loader):
                input = feature.cuda()
                output, _= self.model(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                prob = torch.softmax(_, dim=1)
                results.append({'slide_id': slide_id[0], 'patch_id': patch_id, 'prob': prob[:, 1].cpu().numpy()})

        return results

    def train(self):
        self.model = torch.load("/home/omnisky/sdg/NanTH/zxw/saved_models/35_0.8571428571428571.pth")
        self.test_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.test_loader, False)
        self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.test_loader.dataset.set_mode('selected_bag')

        results = self.inference(self.test_loader)
        for i in range(len(results)):
            slide_path = os.path.join('/home/omnisky/sdg/NanTH/zxw/slide/Microinvasive', results[i]['slide_id'] + '.ndpi')
            try:
                h5_path = os.path.join('/home/omnisky/sdg/NanTH/zxw/cp/Microinvasive/patches', results[i]['slide_id'] + '.h5')
                file = h5py.File(h5_path, "r")
            except:
                continue
            
            coords = file['coords']
            selected_points = []
            for k in range(len(results[i]['patch_id'])):
                selected_points.append(coords[results[i]['patch_id'][k]] / 16)
            
            slide = openslide.OpenSlide(slide_path)
            image = np.array(slide.read_region((0, 0), 4, slide.level_dimensions[4]).convert('RGB'))
            out_image = draw_colored_square(image, selected_points, results[i]['prob'])
            cv2.imwrite(os.path.join('/home/omnisky/sdg/NanTH/zxw/heatmaps/mi', results[i]['slide_id'] + '.png'), out_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
'''