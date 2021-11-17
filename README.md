<h3 align="center">Application of Artificial Neural Networks in COVID-19 and
pneumonia cases diagnosis via CXR images:
A comprehensive analysis of Convolutional Neural Network and
Vision Transformers</h3>

**Abstract.** 
At the end of 2019, humankind was faced with an epidemic—severe acute respiratory syndrome coronavirus 2 (SARS CoV-2) related pneumonia, referred to as coronavirus disease 2019 (COVID-19) that people did not expect to encounter in the current era of technology. Nowadays, the COVID-19 situation becomes more and more severe, as well as sophisticated. And our country, Vietnam is being severely affected by this pandemic. That is why we have to take our actions immediately in order to cope with this pandemic. And the very first step in extinguishing this pandemic is that we have to rapidly and precisely identify COVID-19 cases. One significant improvement of the industrial revolution 4.0 is the application of information technology to medical diagnosis as an automatic method. The advances in artificial intelligence (AI) have enabled the implementation of sophisticated applications that can meet clinical accuracy requirements. Thus, various works propose an AI-based solution via X-ray image diagnosis as a quick testing method, which has high productivity in a short time. They majorly employ the Convolutional neural networks as their base feature extractor module. Especially different from their works, our study proposes the Vision-Transformer-based method as the novel approach. In the experiment, our work covers the comprehensive analysis of both feature extraction approaches and the implementation of an AI diagnosis system for COVID-19 and Viral Pneumonia cases by the state-of-the-art Pyramid Vision Transformer. Along with the strong data augmentation based on the clinical consideration, our best PVTv2-b2-li model achieves **92.99%**, **92.38%** sensitivity and **97.55%**, **89.81%** positive predictive value respectively to the COVID-19 and pneumonia cases on COVIDx8A dataset in the COVID-Net. Our proposed solution can detect COVID-19 in a Chest X-Ray image, that may be a technical proof of the potential of the Transformers-based approach for the vision tasks. The heatmap and confidence score of the detection is also demonstrated, such that the doctors or common users can use them for a final diagnosis in practical usages.

**Keywords**: COVID-19, pneumonia, Deep Learning, Vision Transformer, Chest X-ray (CXR), medical diagnosis.



## A. Paper and Seminar meterial

#### ⭐ For detail of report, watch [this paper](https://github.com/hoangtv2000/COViT/blob/master/paper.pdf).

#### ⭐ For slide of the seminar, watch [here](https://docs.google.com/presentation/d/1jm7SXEqmMi34HPYUbEk2Falys974G5Gi/edit?usp=sharing&ouid=114052551064589379844&rtpof=true&sd=true).

## B. Technial tool

Annotations for modules of the tecnical tool.
+ [config](https://github.com/hoangtv2000/COViT/tree/master/config): Store a configuration file used for both training and inference phase.
+ [data](https://github.com/hoangtv2000/COViT/tree/master/data): Store image and annotation files for training, validation and test, including: all_images (storing all images) folder and train/ val/ test.txt (annotation files). 
+ [dataloader_n_aug](https://github.com/hoangtv2000/COViT/tree/master/dataloader_n_aug): Store code for dataloader and 3 types of data augmentation.
+ [logger](https://github.com/hoangtv2000/COViT/tree/master/logger): Store logger of the training phase.
+ [metrics](https://github.com/hoangtv2000/COViT/tree/master/metrics): Store the specific module for calculating evaluation metric.
+ [model](https://github.com/hoangtv2000/COViT/tree/master/model): Store models, downloaded pretrained checkpoints and modelloader.
+ [predict_n_visualizer](https://github.com/hoangtv2000/COViT/tree/master/predict_n_visualizer): Include code for inference phase and visualizer.
+ [pytorch_grad_cam](https://github.com/hoangtv2000/COViT/tree/master/pytorch_grad_cam): Store Heatmap visualizer by GradCAM, code of this module taken from [here](https://github.com/jacobgil/pytorch-grad-cam).
+ [trainer](https://github.com/hoangtv2000/COViT/tree/master/trainer): Store code for training phase with the ability to track evaluation metrics during this progress.
+ [utils](https://github.com/hoangtv2000/COViT/tree/master/utils): Store utility functions.
+ [TEST notebook](https://github.com/hoangtv2000/COViT/blob/master/TEST.ipynb): Notebook for inference and heatmap visualizer.
+ [TRAIN notebook](https://github.com/hoangtv2000/COViT/blob/master/TRAIN.ipynb): Notebook for training.
+ [alternative_dataloader](https://github.com/hoangtv2000/COViT/tree/master/alternative_dataloader): Store alternative method for dataloader. Replace files which have the same name with files in this folder in these folders above.


## C. Results
### The Exp.5 model
**Click [here](https://drive.google.com/file/d/19parqLAjLeRDvfSzPOuAT9LDHsu8OWbi/view?usp=sharing) to download this model checkpoint.**

Belows are the experimental results of the PVTv2-b2-li in the Exp.5, trained and tested on the COVIDx8A.

#### Experimental results on the test set including 214 COVID-19 test cases.
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
	<em>Heatmap generation and prediction confidence of 8 stochasitcal samples from test set.</em>
</p>
