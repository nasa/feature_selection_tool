# A Data-driven Framework to Select a Cost-Efficient Subset of Parameters to Qualify Sourced Materials

The objective of this code is to develop the process-structure-property linkages and find the optimal set of input variables to qualify the powder feedstock for additively manufactured (AM) Inconel 718 (IN 718). The repository contains the following list of functions developed.

-  Normalizing input variables.
-   Remove low variance variables
-   Calculate and visualize Pearson correlations.
-   Eliminate highly correlated features.
-   Forward Feature Selection.
-   Backward Feature Selection.
-   Sequential floating forward selection.
-   Embedded feature selection with random forest regressor.
-   Embedded feature selection with extra Tree regressor.
-   Check assumptions for regression.
## Requirements

1.  Python 3.5 +
2.  Scikit-Learn
3.  Numpy
4.  Pandas
5.  Matplotlib
6.  Seaborn
7.  Mlxtend

## Dataset
Dataset was collected at NASA Glenn Research Center (Cleveland, OH) and contains measurements of variables from 18 powder lots: 15 pristine and 3 recycled (i.e., used for additive manufacturing a component already) which were produced by either gas or rotary atomization in either Nitrogen or Argon atmosphere. The data types are both categorical and numerical.
