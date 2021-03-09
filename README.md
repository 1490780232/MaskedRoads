# MaskedRoads

**Done:**

* unet

* deeplab_v3

* deeplab_v3_plus

* cal_classes_weight

* focal_loss

* cal_metrics

* commit

* Warmupscheduler

* different scale training

* data transform

* OhemCrossEntropy

* HRnet

  

**Doing:**

* hrnet+ocr Nvidia

**Todo:**

* optimizer
* attention
* refine
* cross_scale_inference
* ......

```
python baseline.py
```

with_all_weights_deep_lab_v3_plus_1024*1024_focal_loss: 0.6551256100225954 [0.42048217 0.8883346 ] 0.9006836579577757

no_weights_deep_lab_v3_plus_1024*1024_focal_loss: 0.7719817341557385 [0.58031979 0.96052724] 0.9639357505458417

no_weights_deep_lab_v3_plus_512*512_focal_loss: 0.8716814212036726 [0.76365912 0.97794872] 0.980642557144165

no_weights_deep_lab_v3_224*224_ce_loss_no_clear: 0.789677 [0.6179563, 0.96043] 0.96396687

no_weights_unet_1024*1024_ce_loss_no_clear: 0.688 [0.4213, 0.949325] 0.952580715