# Transfer Learning Analysis for Diffusion MRI-based Brain Age Prediction Model Generalization

These scripts are used for generalizing the dMRI-based brain age prediction model to new data sources via transfer learning. 
The pretrained model is developed based on the data from [Cam-CAN repository](https://www.cam-can.org/)

## Research Abstract
**Title:** Generalization of diffusion magnetic resonance imaging–based brain age prediction model through transfer learning
**Authors:** Chang-Le Chen<sup>a</sup>, Yung-Chin Hsu<sup>b</sup>, Li-Ying Yang<sup>a</sup>, Yu-Hung Tung<sup>d</sup>, 
Wen-Bin Luo<sup>d</sup>, Chih‐Min Liu<sup>e</sup>, Tzung‐Jeng Hwang<sup>e</sup>, Hai‐Gwo Hwu<sup>e</sup>, Wen-Yih Isaac Tseng<sup>a,c,f</sup>

**Affiliation:**

<sup>a</sup> *Institute of Medical Device and Imaging, College of Medicine, National Taiwan University, Taipei, Taiwan*

<sup>b</sup> *AcroViz Technology Inc., Taipei, Taiwan*

<sup>c</sup> *Graduate Institute of Brain and Mind Sciences, College of Medicine, National Taiwan University, Taipei, Taiwan*

<sup>d</sup> *Department of Medicine, College of Medicine, National Taiwan University, Taipei, Taiwan*

<sup>e</sup> *Department of Psychiatry, National Taiwan University Hospital, Taipei, Taiwan*

<sup>f</sup> *Molecular Imaging Center, National Taiwan University, Taipei, Taiwan*

Brain age prediction models using diffusion magnetic resonance imaging (dMRI) and machine learning techniques enable individual assessment of brain aging status in healthy people and patients with brain disorders. 
However, dMRI data are notorious for high intersite variability, prohibiting direct application of a model to the datasets obtained from other sites. 
In this study, we generalized the dMRI-based brain age model to different dMRI datasets acquired under different imaging conditions. 
Specifically, we adopted a transfer learning approach to achieve domain adaptation. 
To evaluate the performance of transferred models, a brain age prediction model was constructed using a large dMRI dataset as the source domain, and the model was transferred to three target domains with distinct acquisition scenarios. 
The experiments were performed to investigate (1) the tuning data size needed to achieve satisfactory performance for brain age prediction, (2) the feature types suitable for different dMRI acquisition scenarios, and (3) performance of the transfer learning approach compared with the statistical covariate approach. 
By tuning the models with relatively small data size and certain feature types, optimal transferred models were obtained with significantly improved prediction performance in all three target cohorts (p < 0.001). 
The mean absolute error of the predicted age was reduced from 13.89 to 4.78 years in Cohort 1, 8.34 to 5.35 years in Cohort 2, and 8.74 to 5.64 years in Cohort 3. 
The test–retest reliability of the transferred model was validated using dMRI data acquired at two timepoints (intraclass correlation coefficient = 0.950). 
Clinical sensitivity of the brain age prediction model was investigated by estimating the brain age in patients with schizophrenia. 
The prediction made by the transferred model was not significantly different from that made by the reference model. 
Both models predicted significant brain aging in patients with schizophrenia as compared with healthy controls (p < 0.001); the predicted age difference of the transferred model was 4.63 and 0.26 years for patients and controls, respectively, and that of the reference model was 4.39 and -0.09 years, respectively. 
In conclusion, transfer learning approach is an efficient way to generalize the dMRI-based brain age prediction model. 
Appropriate transfer learning approach and suitable tuning data size should be chosen according to different dMRI acquisition scenarios.

## Requirements
If you want to use the current pretrained models we provided, your data have to be analyzed and organized according to the outcomes of [Tract-based Automatic Analysis](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.22854).
Please feel free to get in touch with us if you are interested in this method.

Or, you can use our brain age modeling method to train your regression model based on the data you process on your own way, then using the transfer learning analysis to fine-tune your model adapting to the data from other data.
This modeling framework is not limited to the certain image modality or the task. If you need the technique support, welcome to contact us.




