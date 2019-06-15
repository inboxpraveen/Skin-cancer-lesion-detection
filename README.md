# The HAM10000 Skin Lesion dataset

[LINK TO ORIGINAL DATASET](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)


[ORIGINAL RESEARCH PAPER ON DATASET](https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf)

- The HAM10000 ("Human Against Machine with 10000 training images") dataset is taken from kaggle.


- We collected dermatoscopic images from different populations, acquired and stored by different modalities. The final dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. 


- Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: 
 - Actinic keratoses and intraepithelial carcinoma / Bowen's disease (**akiec**), 
 - basal cell carcinoma (**bcc**), 
 - benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, **bkl**),
 - dermatofibroma (**df**),
 - melanoma (**mel**),
 - melanocytic nevi (**nv**)
 - and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, **vasc**).



- More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by the lesion_id-column within the HAM10000_metadata file.


The model is tested on 2 models. One is a Sequential model and second is a custom resnet model.
The dataset does not have very hard images, hence the sequential model performs better than resnet model. We can also further conclude that the model can also be fine-tuned to perform and get better accuracy score.

### we are able to detect lesion cell type with 75% accuracy. However, the model is a plain vanila network and can be fine-tuned to produce an accuracy score over 80%.
