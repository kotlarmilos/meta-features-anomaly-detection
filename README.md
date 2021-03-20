# Novel Meta-Features for Automated Machine Learning Model Selection in Anomaly Detection

This repository contains the scripts which evaluate a novel set of meta-features for model selection in anomaly detection tasks based on domain-specific properties. 

By using different kinds of metadata, such as the properties of the data, algorithm properties, or
correlation previously derived from the data, it is possible to select different models to effectively solve a
given anomaly detection task. The meta-learning approach based on a set of meta-features that describes
data properties and correlation can enable efficient model selection in AutoML frameworks.

Experiments with 63 datasets from different repositories with varying schemas show that
the proposed set of meta-features achieves the accuracy of 87% for model selection, while the achieved
accuracy for simple meta-features is 74%, for statistical meta-features 68%, for information theory metafeature
70%, and for a comprehensive set of meta-features by [pyMFE](https://pypi.org/project/pymfe/) 73%.

Results are in [/results](https://github.com/kotlarmilos/meta-features-anomaly-detection/tree/master/pycharm/results) directory. Evaluated algorithms are in [/algorithms](https://github.com/kotlarmilos/meta-features-anomaly-detection/tree/master/pycharm/algorithms) directory.

## Datasets

Datasets are collected from repositories in [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OPQMVF) and [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) and cover a broad range of domains including manufacturing, transportation, healthcare, intrusion detection, and system log analysis.

## Prerequests

 - Python 3.7
 - Pip
 - Numpy
