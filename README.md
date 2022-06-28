# Toward Generalizable Cloud Resource Forecasting Using an Image-based Machine Learning Pipeline
## Abstract
Efficient cloud resource management, provisioning and scheduling requires mechanisms to predict future resource utilization at the workload-, task-, job-, node- and cluster-level in an online, adaptive and practical manner. Recent solutions leverage machine learning methods to learn patterns of resource utilization, adapting to different observed behaviors via training different models or performing online training. It is still very challenging to train a single model with robust generalization capabilities, that accurately forecasts unseen data of potentially different patterns without the need for re-training. This vision paper makes a case as to how we can build generalizable prediction models to learn patterns of cloud resource usage using a purely image-based system pipeline, inspired by the strong generalizable capabilities of image-based machine learning methods. Our analysis shows that the proper visualization of cloud resource consumption data, coupled with image-based machine learning methods similar to those used in video frame prediction, is able to produce highly accurate forecasts that are generalizable to unseen data with very different trends from the training dataset. Our vision is to use such image-based forecasting components to build an online, adaptive and generalizable cloud resource management systems. This work aims to highlight the potential and lay the foundations for future use of computer vision and images in system-level solutions.

## Requirements
Using a virtual environment is recommended <br>

Clone the repository and run to install all required libraries:

```
pip install -r requirements.txt
```


## DataExploration
Data exploration, data preprocessing and clustering.

[Dataset exploration](DataExploration/README.md)


Figures: [Figures Dataset exploration](Figures/DataExploration/README.md)

## Modeling
Modeling and evaluation.

[Modeling](Modeling/README.md)




Contact
-------------------------------------
- [Javier Galindos](https://www.linkedin.com/in/javiergalindos/): javier.galindos [at] imdea.org

Acknowledgments
-----------------
- [Thaleia Doudali](https://thaleia-dimitradoudali.github.io) (IMDEA Software)


