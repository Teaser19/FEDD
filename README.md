# Feature Extraction for Explicit Concept Drift Detection (FEDD) in time series

Python implementation of FEDD drift detector according to the concept and logic introduced in: 
Cavalcante, R. C., Minku, L. L., & Oliveira, A. L. I. (2016). FEDD: Feature Extraction for Explicit Concept Drift Detection in time series. In 2016 International Joint Conference on Neural Networks (IJCNN): 24-29 July 2016, Vancouver, Canada (pp. 740–747). Piscataway, NJ: IEEE. https://doi.org/10.1109/IJCNN.2016.7727274

Implementation of ECDD drift detector is based on:
Ross, G. J., Adams, N. M., Tasoulis, D. K., & Hand, D. J. (2012). Exponentially weighted moving average charts for detecting concept drift. Pattern Recognition Letters, 33(2), 191–198. https://doi.org/10.1016/j.patrec.2011.08.019

Implementation of the bicorrelation and mutual information features is done corresponding to:
Dimitris Kugiumtzis (2020). Measures of Analysis of Time Series toolkit (MATS) (https://www.mathworks.com/matlabcentral/fileexchange/27561-measures-of-analysis-of-time-series-toolkit-mats), MATLAB Central File Exchange

The structure of FEDD corresponds to the scikit-multiflow implementation for drift detection (https://github.com/scikit-multiflow/scikit-multiflow/tree/a7e316d1cc79988a6df40da35312e00f6c4eabb2/src/skmultiflow/drift_detection) and can be used analogously.
