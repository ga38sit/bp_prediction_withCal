This Git Repository contains our used models of our work "The Role of Dataset Integrity, Calibration and Signal Quality in Neural Network-Based Blood Pressure Estimation" as .py files, as well as the used models from [1] and [2] as our benchmark test.

Our Models:
- mlp_withCal 		--> MLP processing only scalar features model with flag, wether to include calibration data or not
- resnet_withCal 	--> added ResNet based block for processing signal inputs parallel to the MLP, processing the scalar features. Also includes flag, which enables/disables the usage of calibration data
- tcn_withCal 		--> added TCN based block for processing signal inputs parallel to the MLP, processing the scalar features. Also includes flag, which enables/disables the usage of calibration data

Benchmark Models: 
- mlp_benchmark 			--> Model from [1], adjusted to use our predictions heads for DBP, SBP and PP
- deep_bp_benchmark 	--> Model from [2], adjusted to use our predictions heads for DBP, SBP and PP



[1] Hsu, Y.-C., Li, Y.-H., Chang, C.-C., & Harfiya, L. N. (2020). Generalized Deep Neural Network Model for Cuffless Blood Pressure Estimation with Photoplethysmogram Signal Only. Sensors, 20(19), 5668. https://doi.org/10.3390/s20195668

[2] C. Yan et al., "Novel Deep Convolutional Neural Network for Cuff-less Blood Pressure Measurement Using ECG and PPG Signals," 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Berlin, Germany, 2019, pp. 1917-1920, doi: 10.1109/EMBC.2019.8857108
