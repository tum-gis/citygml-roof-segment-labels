# citygml-roof-segment-labels

Generate datasets of roof segment labels for aerial imagery derived from CityGML semantic 3D city models for semantic segmentation as described in [1].


## Overview

The notebook _dataset\_creation.ipynb_ provides steps and detailed instructions for the dataset creation (data pre-processing, generation of image and label files, data split). The roof segment geometries for all configurations are available in the subdirectory _segments_. The locations selected for validation and test sets are located in _val\_test\_locations_. To create custom datasets using the code provided here, make sure your data is structured identically to the data in these subdirectories.

The roof segment geometries for the configuration _small-manu_ are sourced from [2].


## References

[1] Faltermeier, F.L.; Krapf, S.; Willenborg, B.; Kolbe, T.H. (2023): Improving Semantic Segmentation of Roof Segments Using Large-Scale Datasets Derived from 3D City Models and High-Resolution Aerial Imagery. Remote Sens. 2023, 15, 1931. https://doi.org/10.3390/rs15071931

[2] Krapf, S.; Bogenrieder, L.; Netzler, F.; Balke, G.; Lienkamp, M. (2022): RID—Roof Information Dataset for Computer Vision-Based Photovoltaic Potential Assessment. Remote Sens. 2022, 14, 2299. https://doi.org/10.3390/rs14102299
