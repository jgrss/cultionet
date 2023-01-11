# Changelog

<!--next-version-placeholder-->

## v1.6.2 (2023-01-11)
* Addressed security issue by bumping `setuptools` ([#42](https://github.com/jgrss/cultionet/pull/42))

## v1.6.1 (2023-01-04)
* Reorganized CLI arguments ([#41](https://github.com/jgrss/cultionet/pull/41))

## v1.6.0 (2023-01-03)
* ([#40](https://github.com/jgrss/cultionet/pull/40))
* New architecture design based on UNet 3+ and residual convolutions
* Modified total loss quantification with deep supervision of crop type in RNN layer
* Added `num_workers` option in `DataLoader` for faster train/predict
* Added .pt data compression by changing `torch.save|load` to `joblib.dump|load`

## v1.5.2 (2022-11-10)
* Fixed in-memory predictions ([#38](https://github.com/jgrss/cultionet/pull/38))

## v1.5.1 (2022-11-10)
* Fixes model outputs during serial predictions ([#37](https://github.com/jgrss/cultionet/pull/37))

## v1.5.0 (2022-11-9)
*  PR [#36](https://github.com/jgrss/cultionet/pull/36)
*  Re-configures the model architecture to improve the distance transform predictions
*  Improved prediction performance through pytorch-lightning batch predictions
*  Improves the edge calculations during cultionet create by using geometry boundary
*  Adds improved performance for mean and standard deviation calculations
*  Adds experimental orientation
*  Adds experimental Mask-RCNN
*  Adds a Docker file for CUDA 11.6

## v1.4.0 (2022-10-24)
* Added spatial k-fold cross-validation ([#35](https://github.com/jgrss/cultionet/pull/35))

## v1.3.3 (2022-10-19)
* Re-added serial predictions over CPU or GPU ([#34](https://github.com/jgrss/cultionet/pull/34))

## v1.3.2 (2022-10-12)
* Added default values for gain and offset

## v1.3.1 (2022-10-12)
* Added YAML files to distributed package data

## v1.3.0 (2022-10-11)
* Added a test dataset CLI option, additional model callbacks, and faster pre-model dimension checking ([#33](https://github.com/jgrss/cultionet/pull/33))

## v1.2.3 (2022-10-04)
* Added user argument to specify raster compression method ([#32](https://github.com/jgrss/cultionet/pull/32))

## v1.2.2 (2022-10-03)
* Updated geowombat version ([#30](https://github.com/jgrss/cultionet/pull/30))

## v1.2.1 (2022-10-01)
* Add concurrent predictions over image windows ([#29](https://github.com/jgrss/cultionet/pull/29))

## v1.1.1 (2022-09-16)
* Add argument for weight decay ([#20](https://github.com/jgrss/cultionet/pull/20))

## v1.1.0 (2022-01-14)
* Add config argument to specify region ids as a file ([#28](https://github.com/jgrss/cultionet/pull/28))

## v1.0.1 (2022-06-18)
* Fix band and time dimension getting with single-band images
 
## v1.0.0 (2022-06-12)
* Increase cultionet public version
