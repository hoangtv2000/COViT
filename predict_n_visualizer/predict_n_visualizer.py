import torch, torchvision
import numpy as np
import torch.nn.functional as F
import pandas as pd
from dataloader_n_aug.dataloader import get_test_data, get_val_data

from metrics.metric import MetricTracker, sensitivity, positive_predictive_value, F1_macro_score
from model.modelloader import load_checkpoint
from utils.util import *

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import matplotlib.pyplot as plt
import seaborn as sn

from PIL import Image
from dataloader_n_aug.image_aug import *
from torchvision import transforms


class PredictAndVisualizer:
    """Class for predict on test_set and visualization.
    """
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
        self.class_names = self.config.dataset.class_dict.keys()

        if (self.config.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")


        self.test_metric = MetricTracker('acc', mode='validation')


        print('----- LOADING PRETRAINED CHECKPOINTS -----')
        # Get model and checkpoint
        get_checkpoints(self.model_name)
        self.model = get_model(self.config, self.model_name)

        # Load model and add model to device
        checkpoint_name = input("Choose one of these checkpoints above: ")
        cpkt_fol_name = os.path.join(self.config.cwd, f'checkpoints/{self.model_name}/{checkpoint_name}')
        self.checkpoint_dirmodel = f'{cpkt_fol_name}/model_best_checkpoint.pth'
        self.model, _, _, _ = load_checkpoint(self.checkpoint_dirmodel, self.model)
        self.model = self.model.to(self.device)
        self.model.eval()





    def predict(self):
        # Get data
        test_data_loader = get_test_data(self.config)

        confusion_matrix = torch.zeros(self.config.dataset.num_classes,\
                                    self.config.dataset.num_classes)

        self.test_metric.reset()

        results = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_data_loader):
                data = data.to(self.device)
                gr_truth = target.clone().detach().to(self.device)
                output = self.model(data)

                prediction = torch.argmax(output.clone().detach(), dim=1)

                accuracy = np.sum(prediction.cpu().numpy() == target.cpu().numpy())

                writer_step = len(test_data_loader) + batch_idx
                self.test_metric.update(key='acc',
                                          value = accuracy,
                                          n=target.size(0), writer_step=writer_step)

                for tar, pred in zip(target.cpu().view(-1), prediction.cpu().view(-1)):
                    confusion_matrix[tar.long(), pred.long()] += 1

        # Metrics
        s = sensitivity(confusion_matrix.numpy())
        ppv = positive_predictive_value(confusion_matrix.numpy())
        acc = self.test_metric.avg('acc')
        F1_score = F1_macro_score(confusion_matrix.numpy())

        print('-'*30)
        print(f'OVERALL Accuracy:     {acc}')
        print(f'OVERALL Sensivity:    {s}')
        print(f'OVERALL PosPredValue: {ppv}')
        print(f'Macro_avg F1-score:   {F1_score}')
        print('-'*30)

        return confusion_matrix





    def plot_confusion_matrix(self, confusion_matrix):
        cf_df = pd.DataFrame(confusion_matrix, index = [i for i in self.class_names],
                  columns = [i for i in self.class_names])

        sn.heatmap(cf_df, annot=True, cmap='Blues', fmt='g')





    def visualize(self, num_samples=9):
        """Visualize heat map by gradients.
        """
        assert (self.model_name == 'model_PVT_V2'), 'Only support model_PVT_V2_B2_Linear!'
        assert num_samples <= self.config.dataloader.test.batch_size, '"num_samples" must <= "batch_size"'

        # Get test data for Gradcam visualizer
        test_data_loader = get_test_data(self.config, resize4gradcam=True)

        fig = plt.figure(figsize=(120, 120))

        cam = GradCAMPlusPlus(model=self.model,
                target_layers= [self.model.pvt.block4[1].norm1],
                use_cuda=True,
                reshape_transform=self.reshape_transform)

        (batch_data, batch_target) = next(iter(test_data_loader))

        counter=0
        for (data, target) in zip(batch_data[:num_samples], batch_target[:num_samples]):
            data = data.unsqueeze(0).to(self.device)

            # Cam visualize
            grayscale_cam = cam(input_tensor=data, #FF 12 times
                            target_category=None,
                            eigen_smooth=True,
                            aug_smooth=True)[0, :]

            rgb_img = augmentation2raw(self.config, data.squeeze(0).cpu())
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)

            # predict
            output = self.model(data)
            percent_pred = F.softmax(output).cpu()
            prediction = int(torch.argmax(output, dim=1).cpu())


            _, axarr = plt.subplots(1,2)

            counter+=1
            axarr[0].set_title(\
                'Image: {:>4}\n Ground truth: {:>4}\n Model prediction: {:>4}'\
                .format(counter, list(self.class_names)[int(target)], list(self.class_names)[prediction]))

            percent_out = list(percent_pred.detach().numpy())[0]*100

            axarr[1].set_title(\
                'Class confidence:\n Pneumonia : {:.2f}%\n Normal : {:.2f}%\n COVID_19 : {:.2f}%'\
                .format(percent_out[0], percent_out[1], percent_out[2]))

            axarr[0].imshow(rgb_img)
            axarr[1].imshow(cam_image)

            axarr[0].axis('off')
            axarr[1].axis('off')





    def inference(self, img_path):
        """Inference with an any image.
        """
        img = Image.open(img_path).convert('RGB')
        infer_img = img.resize(self.config.dataset.gradcam_img_size)

        fig = plt.figure(figsize=(120, 120))

        cam = GradCAMPlusPlus(model=self.model,
                target_layers= [self.model.pvt.block4[1].norm1],
                use_cuda=True,
                reshape_transform=self.reshape_transform)

        if self.config.preprocess_type == 'base' or self.config.preprocess_type == 'autoaug':
            transform = base_test_transformation()
            processed_img = transform(infer_img).unsqueeze(0).to(self.device)

        elif self.config.preprocess_type == 'torchio':
            convert_tensor = transforms.ToTensor()
            converted_img = convert_tensor(infer_img).unsqueeze(3)

            transform = torchio_test_transformation()
            processed_img = transform(converted_img).squeeze(3).to(self.device)

        # Cam visualize
        grayscale_cam = cam(input_tensor=processed_img, #FF 12 times
                        target_category=None,
                        eigen_smooth=True,
                        aug_smooth=True)[0, :]

        # predict
        output = self.model(processed_img)
        percent_pred = F.softmax(output).cpu()
        prediction = int(torch.argmax(output, dim=1).cpu())

        percent_out = list(percent_pred.detach().numpy())[0]*100

        rgb_img = augmentation2raw(self.config, processed_img.squeeze(0).cpu())
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)


        print('Model prediction: {:>4}'.format(list(self.class_names)[prediction]))

        print('Class confidence:\n\
             Pneumonia : {:.2f}%\n\
             Normal : {:.2f}%\n\
             COVID_19 : {:.2f}%'.format(percent_out[0], percent_out[1], percent_out[2]))


        _, axarr = plt.subplots(1,2)
        axarr[0].imshow(infer_img)
        axarr[1].imshow(cam_image)

        axarr[0].axis('off')
        axarr[1].axis('off')


    def reshape_transform(self, tensor, height=7, width=7):
        # input: torch.Size([1, 49, 512])
        # target: torch.Size([1, 512, 7, 7])
        result = tensor.reshape(tensor.size(0),
                                height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result
