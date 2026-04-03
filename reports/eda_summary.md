# EDA Summary

## Data Contract
- Dataset path: `data/Данные_для_курсовои_Классическое_МО.xlsx`
- SHA256: `87fabec4e77159df7482ca480236712c031500d3935141b3231368d83a67c659`
- Rows: `1001`
- Columns: `214`
- Descriptor features after leakage guard: `210`
- Missing values: `36`
- Duplicates: `0`

## Main Findings
- Все три regression targets положительные и сильно skewed: IC50=`3.675`, CC50=`1.973`, SI=`18.013`.
- Пропуски сосредоточены в `12` дескрипторах и составляют `36` значений суммарно.
- Median-based classification задачи почти идеально сбалансированы; `SI > 8` заметно более дисбалансна.
- Корреляция `IC50` и `CC50` умеренная положительная, а `SI` с ними линейно связан слабо из-за ratio-природы и тяжёлого хвоста.

## Top Descriptor Correlations
- IC50, mM: VSA_EState4 (-0.274), Chi2n (-0.257), PEOE_VSA7 (-0.256), Chi2v (-0.249), fr_Ar_NH (0.246)
- CC50, mM: MolMR (-0.310), LabuteASA (-0.309), MolWt (-0.306), ExactMolWt (-0.306), HeavyAtomCount (-0.305)
- SI: BalabanJ (0.163), fr_NH2 (0.160), RingCount (-0.124), fr_Al_COO (0.102), fr_COO (0.101)

## Saved Artifacts
- `/Users/davidsukhashvili/Desktop/ML/MEPhi/ClassicMl/Neural Networks in Chemistry/results/eda/target_distributions.png`
- `/Users/davidsukhashvili/Desktop/ML/MEPhi/ClassicMl/Neural Networks in Chemistry/results/eda/target_boxplots.png`
- `/Users/davidsukhashvili/Desktop/ML/MEPhi/ClassicMl/Neural Networks in Chemistry/results/eda/target_correlations.png`
- `/Users/davidsukhashvili/Desktop/ML/MEPhi/ClassicMl/Neural Networks in Chemistry/results/eda/missing_values.csv`
- `/Users/davidsukhashvili/Desktop/ML/MEPhi/ClassicMl/Neural Networks in Chemistry/results/eda/top_descriptor_target_correlations.csv`