<h3 align="center">Application of Artificial Neural Networks in COVID-19 and
pneumonia cases diagnosis via CXR images:
A comprehensive analysis of Convolutional Neural Network and
Vision Transformers</h3>

## A. Theoretical comprehensive analysis of the CNN and ViT

## B. Techincal tool for COVID-19 and Pneumonia cases diagnosis and heatmap generation  

+ [config](https://github.com/hoangtv2000/COViT/tree/master/config): Store a configuration file used for both training and inference phase.
+ [data](https://github.com/hoangtv2000/COViT/tree/master/data): Store annotation files for training, validation and test, including: train/ val/ test.txt
+ [dataloader_n_aug](https://github.com/hoangtv2000/COViT/tree/master/dataloader_n_aug): Store code for dataloader and 3 types of data augmentation.
+ [logger](https://github.com/hoangtv2000/COViT/tree/master/logger): Store logger of the training phase.
+ [metrics](https://github.com/hoangtv2000/COViT/tree/master/metrics): Store the specific module for calculating evaluation metric.
+ [model](https://github.com/hoangtv2000/COViT/tree/master/model): Store models, downloaded pretrained checkpoints and modelloader.
+ [predict_n_visualizer](https://github.com/hoangtv2000/COViT/tree/master/predict_n_visualizer): Include code for inference phase and visualizer.
+ [pytorch_grad_cam](https://github.com/hoangtv2000/COViT/tree/master/pytorch_grad_cam): Store Heatmap visualizer by GradCAM, code of this module taken from [here](https://github.com/jacobgil/pytorch-grad-cam)
+ [trainer](https://github.com/hoangtv2000/COViT/tree/master/trainer): Store code for training phase with the ability to track evaluation metrics during this progress.
+ [utils](https://github.com/hoangtv2000/COViT/tree/master/utils): Store utility functions.
+ [TEST notebook](https://github.com/hoangtv2000/COViT/blob/master/TEST.ipynb): Notebook for inference and heatmap visualizer.
+ [TRAIN notebook](https://github.com/hoangtv2000/COViT/blob/master/TRAIN.ipynb): Notebook for training.
+ [alternative_dataloader](https://github.com/hoangtv2000/COViT/tree/master/alternative_dataloader): Store alternative method for dataloader. Replace files which have the same name with files in this folder in these folders above.


## C. Results
### Model Exp.5 (PVTv2-b2-li) in COVIDx8A (214 COVID-19 test)
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">92.38</td>
    <td class="tg-c3ow">93.0</td>
    <td class="tg-c3ow">92.99</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">PPV (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">89.81</td>
    <td class="tg-c3ow">86.91</td>
    <td class="tg-c3ow">97.55</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <td class="tg-7btt">Overall Acc. (%)</td>
    <td class="tg-7btt">Macro Avg. F1 (%)</td>
  </tr>
  <tr>
    <td class="tg-c3ow">92.84</td>
    <td class="tg-c3ow">92.05</td>
  </tr>
</table></div>


### Heatmap Genearation
<p align="center">
	<img src="https://github.com/hoangtv2000/COViT/blob/master/pics/ss.png" alt="photo not available" width="100%" height="100%">
	<br>
	<em>Heatmap generation and prediction confidence of 9 samples from test set.</em>
</p>
