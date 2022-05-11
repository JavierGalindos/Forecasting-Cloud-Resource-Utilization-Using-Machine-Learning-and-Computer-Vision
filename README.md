# Forecasting cloud resource utilization using Machine Learning and Computer Vision
## Abstract
Cloud computing has revolutionized the access and use of computing hardware technologies at a massive scale. The cloud computing paradigm, despite its enormous success and numerous benefits, faces several obstacles. A major challenge is resource provisioning for computational tasks. There is a need to make accurate forecasts of resource utilization to achieve efficient resource management and cost efficiency in cloud environments. Cloud resource utilization traces are rather complex and random. Traditional time series forecasting methods are not able to capture the non-linearity and complexity of cloud resource utilization. In recent times, several machine learning approaches have attempted to solve this problem using more sophisticated models, but fail to provide highly accurate predictions in return for low training and inference times. Recent work in the financial domain shows how the use of an image representation of time series data, and relevant image-based methods can lead to more robust and effective forecasting. To this end, this thesis explores the use of images, computer vision and machine learning methods to forecast future resource utilization in cloud environments. An image-based prediction pipeline is proposed, that visualizes data in a sophisticated way, uses image-based machine learning methods to predict resource consumption, similar to those used in video frame prediction, and then decomposes the predicted images back to numeric predictions. Furthermore, this thesis includes an in-depth comparative analysis that shows how an image-based prediction pipeline can provide accurate forecasts for long windows of time in the future, as well as capture the short-term patterns and overall trends of the data. Most importantly, the proposed image-based machine learning method, typically used in video frame prediction, can accurately forecast resource utilization, even when the training and inference datasets exhibit completely different characteristics, something that is currently not possible using other image-based, non-image-based and traditional forecasting methods.

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


