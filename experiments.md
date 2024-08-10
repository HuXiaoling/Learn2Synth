## Print the parameters
```
model['state_dict']['network.synthnet.sigma'], torch.sigmoid(model['state_dict']['network.synthnet.weight_low']), 
torch.sigmoid(model['state_dict']['network.synthnet.weight_middle']), torch.sigmoid(model['state_dict']['network.synthnet.weight_high'])
```

## Experiemnt1: only noise

### Summary: Training on different noise levels and testing on different noise levels as well

(Low bound?) Row is different models, column is different level noise images

| Noise-level | noise-free (SynthSeg) | noise-free (LearnableSynthSeg) | 0.05 | 0.1 | 0.15 | vary | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| noise-free    | 0.9127    | 0.9231    | 0.8995    | 0.8741    | 0.8602    | 0.8661    |
| 0.05          | 0.3100    | 0.3459    | 0.8866    | 0.8730    | 0.8551    | 0.8637    |
| 0.1           | 0.2092    | 0.1717    | 0.7436    | 0.8584    | 0.8540    | 0.8506    |
| 0.15          | 0.1423    | 0.0918    | 0.4917    | 0.7994    | 0.8396    | 0.8332    |
| vary          | 0.2045    | 0.1505    | 0.6353    | 0.8310    | 0.8336    | 0.8386    |

#### Learn2Synth: Different noise levels

| Setting | Preset Sigma | Opmitized Sigma | File | States |
| :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | Fixed | train_0.sh                    | Done  |
| Learn2Synth   | 0.05          | Fixed | train_005.sh  (experiment_12) | Done  |
| Learn2Synth   | 0.1           | Fixed | train_010.sh  (experiment_5)  | Done  |
| Learn2Synth   | 0.15          | Fixed | train_015.sh  (experiment_13) | Done  |
| Learn2Synth   | [0.025, 0.2]  | Fixed | train_vary.sh (experiment_15) | Done  |
| Learn2Synth   | [0.025, 0.2]  | Range | train_vary_vary.sh            | Done  |

#### SynthSeg: Different noise levels

| Setting | Preset Sigma | File | States |
| :----: | :----: | :----: | :----: |
| SynthSeg  | 0             | train_free_ori.sh     | Done  |
| SynthSeg  | 0.05          | train_005_free.sh     | Done  |
| SynthSeg  | 0.1           | train_010_free.sh     | Done  |
| SynthSeg  | 0.15          | train_015_free.sh     | Done  |
| SynthSeg  | [0.025, 0.2]  | train_vary_free.sh    | Done  |

#### SynthSeg: Different noise levels by using the parameters learned by Learn2Synth

| Setting | Preset Sigma | File | States |
| :----: | :----: | :----: | :----: |
| SynthSeg  | -0.0023           | train_synthseg_free_learned.sh        | Done  |
| SynthSeg  | 0.0402            | train_synthseg_005_learned.sh         | Done  |
| SynthSeg  | 0.0993            | train_synthseg_010_learned.sh         | Done  |
| SynthSeg  | 0.1466            | train_synthseg_015_learned.sh         | Done  |
| SynthSeg  | 0.1328            | train_synthseg_vary_learned.sh        | Done  |
| SynthSeg  | [0.0984, 0.1525]  | train_synthseg_vary_vary_learned.sh   | Done  |

#### Learn2Synth: Results for $\sigma = 0$

| Pre-set | sigma = 0 | Dice |
| :----: | :----: | :----: |
| model1        | 0.0004    | 0.889 |
| model2        | -0.0009   | 0.897 |
| model3        | -0.0008   | 0.897 |
| model4        | -0.0014   | 0.897 |
| **model5**    | -0.0023   | 0.898 |

#### Learn2Synth: Results for $\sigma = 0.05$

| Pre-set | sigma = 0.05 | Dice |
| :----: | :----: | :----: |
| model1        | 0.0411    | 0.886 |
| model2        | 0.0413    | 0.877 |
| **model3**    | 0.0402    | 0.893 |
| model4        | 0.0435    | 0.877 |
| model5        | 0.0440    | 0.879 |

#### Learn2Synth: Results for $\sigma = 0.1$

| Pre-set | sigma = 0.1 | Dice |
| :----: | :----: | :----: |
| model1        | 0.0958    | 0.853 |
| **model2**    | 0.0993    | 0.868 |
| model3        | 0.0981    | 0.854 |
| model4        | 0.0971    | 0.859 |
| model5        | 0.0977    | 0.852 |

#### Learn2Synth: Results for $\sigma = 0.15$

| Pre-set | sigma = 0.15 | Dice |
| :----: | :----: | :----: |
| model1        | 0.1446    | 0.840 |
| model2        | 0.1443    | 0.823 |
| model3        | 0.1457    | 0.829 |
| **model4**    | 0.1466    | 0.850 |
| model5        | 0.1482    | 0.823 |

#### Learn2Synth: Results for $\sigma = [0.025, 0.2]$

| Pre-set | sigma = [0.025, 0.2] | Dice |
| :----: | :----: | :----: |
| **model1**    | 0.1328    | 0.841 |
| model2        | 0.1377    | 0.822 |
| model3        | 0.1320    | 0.826 |
| model4        | 0.1318    | 0.830 |
| model5        | 0.1376    | 0.840 |

#### Learn2Synth: Results for $\sigma = [0.025, 0.2]$ (regress ranges of sigma)

| Pre-set | sigma = [0.025, 0.2] | Dice |
| :----: | :----: | :----: |
| model1        | [0.1008, 0.1522]  | 0.825 |
| model2        | [0.0979, 0.1523]  | 0.837 |
| **model3**    | [0.0984, 0.1525]  | 0.839 |
| model4        | [0.0995, 0.1531]  | 0.838 |
| model5        | [0.0986, 0.1507]  | 0.830 |

#### Learn2Synth: Results for fine-scale sigma settings

| Pre-set sigma | 0.05 | 0.09 | 0.095 | 0.1 | 0.105 | 0.11 | 0.15 | vary |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| model1    | 0.0411    | 0.0829    | 0.0930    | 0.0958    | 0.0997    | 0.1069    | 0.1446    | 0.1328    |
| model2    | 0.0413    | 0.0816    | 0.0925    | 0.0993    | 0.0990    | 0.1040    | 0.1443    | 0.1377    |
| model3    | 0.0402    | 0.0849    | 0.0938    | 0.0981    | 0.0975    | 0.1076    | 0.1457    | 0.1320    |
| model4    | 0.0435    | 0.0858    | 0.0918    | 0.0971    | 0.1010    | 0.1056    | 0.1466    | 0.1318    |
| model5    | 0.0440    | 0.0851    | 0.0934    | 0.0977    | 0.0993    | 0.1067    | 0.1482    | 0.1376    |

| Pre-set sigma | 0.095 | 0.096 | 0.097 | 0.098 | 0.099 | 0.1 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| model1    | 0.0934/0.8507 | 0.0900/0.8532 | 0.0931/0.8694 | 0.0942/0.8755 | 0.0935/0.8745 | 0.0958/0.8502 |
| model2    | 0.0938/0.8607 | 0.0891/0.8689 | 0.0939/0.8533 | 0.0946/0.8666 | 0.0946/0.8521 | 0.0993/0.8560 |
| model3    | 0.0951/0.8767 | 0.0882/0.8744 | 0.0909/0.8600 | 0.0944/0.8629 | 0.0942/0.8528 | 0.0981/0.8555 |
| model4    | 0.0945/0.8584 | 0.0899/0.8567 | 0.0913/0.8604 | 0.0948/0.8673 | 0.0941/0.8616 | 0.0971/0.8677 |
| model5    | 0.0933/0.8590 | 0.0897/0.8697 | 0.0929/0.8620 | 0.0965/0.8705 | 0.0983/0.8741 | 0.0977/0.8630 |

## Experiemnt2: both noise and bias field
```
synthetic real images = noise_free_image * bias filed + noise

Bias field = ((2 * 2 -> upsampling to 256 * 256) ** (eps_a * a) * ((4 * 4 -> upsampling to 256 * 256) 
** (eps_b * b)) * ((8 * 8 -> upsampling to 256 * 256) ** (eps_c * c))
```

#### Learn2Synth: Results for $\sigma = 0.05$, $c_{low} = 1$, $c_{middle} = 0$, $c_{high} = 0$

| Pre-set | sigma = 0.05 | low = 1/middle = 0/high = 0 | Dice |
| :----: | :----: | :----: | :----: |
| model1    | 0.0508    | 0.1977/0.2408/0.5661    | 0.8406    |
| model2    | 0.0547    | 0.1615/0.2161/0.5581    | 0.8514    |
| model3    | 0.0538    | 0.1392/0.2050/0.5529    | 0.8388    |
| model4    | 0.0535    | 0.1377/0.2022/0.5514    | 0.8548    |
| model5    | 0.0560    | 0.1334/0.1994/0.5519    | 0.8515    |

#### Learn2Synth: Results for $\sigma = 0.1$, $c_{low} = 1$, $c_{middle} = 0$, $c_{high} = 0$

| Pre-set | sigma = 0.10 | low = 1/middle = 0/high = 0 | Dice |
| :----: | :----: | :----: | :----: |
| model1    | 0.0917    | 0.1266/0.1835/0.5389    | 0.8362    |
| model2    | 0.0918    | 0.1220/0.1832/0.5391    | 0.8356    |
| model3    | 0.0920    | 0.1214/0.1833/0.5386    | 0.8300    |
| model4    | 0.0961    | 0.1083/0.1750/0.5321    | 0.8290    |
| model5    | 0.0981    | 0.0882/0.1581/0.5201    | 0.8180    |

#### Learn2Synth: Results for $\sigma = 0.15$, $c_{low} = 1$, $c_{middle} = 0$, $c_{high} = 0$

| Pre-set | sigma = 0.15 | low = 1/middle = 0/high = 0 | Dice |
| :----: | :----: | :----: | :----: |
| model1    | 0.1336    | 0.1166/0.1822/0.5207    | 0.8109    |
| model2    | 0.1361    | 0.1097/0.1794/0.5168    | 0.8096    |
| model3    | 0.1370    | 0.1040/0.1750/0.5163    | 0.8240    |
| model4    | 0.1389    | 0.0992/0.1721/0.5150    | 0.8276    |
| model5    | 0.1402    | 0.0923/0.1662/0.5090    | 0.8254    |

#### Check if $\sigma$ always below the pre-set value

| Model | Sigma | Sigma + 0.2 |
| :----: | :----: | :----: |
| model1    | -0.1055   | 0.0945    |
| model2    | -0.1034   | 0.0966    |
| model3    | -0.1036   | 0.0964    |
| model4    | -0.1046   | 0.0954    |
| model5    | -0.1043   | 0.0957    |

#### Learn2Synth: Results for $\sigma = 0.05$, $c_{low} = 0.5$

| Pre-set | sigma = 0.05 | low = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0498    | 0.4947    | 
| model2    | 0.0506    | 0.4945    |
| model3    | 0.0511    | 0.4922    |
| model4    | 0.0480    | 0.4901    |
| model5    | 0.0489    | 0.4890    |

#### Learn2Synth: Results for $\sigma = 0.05$, $c_{middle} = 0.5$

| Pre-set | sigma = 0.05 | middle = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0490    | 0.5855    | 
| model2    | 0.0486    | 0.5853    |
| model3    | 0.0485    | 0.5853    |
| model4    | 0.0482    | 0.5852    |
| model5    | 0.0480    | 0.5854    |

#### Learn2Synth: Results for $\sigma = 0.05$, $c_{high} = 0.5$

| Pre-set | sigma = 0.05 | high = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0501    | 0.5742    | 
| model2    | 0.0492    | 0.5738    |
| model3    | 0.0506    | 0.5744    |
| model4    | 0.0511    | 0.5745    |
| model5    | 0.0503    | 0.5740    |

#### Learn2Synth: Results for $\sigma = 0.1$, $c_{low} = 0.5$

| Pre-set | sigma = 0.1 | low = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0946    | 0.5996    | 
| model2    | 0.0972    | 0.5989    |
| model3    | 0.0962    | 0.5981    |
| model4    | 0.0984    | 0.5965    |
| model5    | 0.0985    | 0.5964    |

#### Learn2Synth: Results for $\sigma = 0.1$, $c_{middle} = 0.5$

| Pre-set | sigma = 0.1 | middle = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.1016    | 0.5767    | 
| model2    | 0.1044    | 0.5773    |
| model3    | 0.1031    | 0.5767    |
| model4    | 0.1036    | 0.5762    |
| model5    | 0.1042    | 0.5761    |

#### Learn2Synth: Results for $\sigma = 0.1$, $c_{high} = 0.5$

| Pre-set | sigma = 0.1 | high = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0992    | 0.5991    | 
| model2    | 0.0994    | 0.5995    |
| model3    | 0.1005    | 0.6005    |
| model4    | 0.0991    | 0.6013    |
| model5    | 0.0994    | 0.6014    |

### Summary: Training on different noise levels and testing on different noise levels as well

| Noise-level | noise-free (LearnableSynthSeg)  | sigma = 0.05  | sigma = 0.1 | sigma = 0.15    | sigma = [0.025, 0.2] |
| :----: | :----: | :----: | :----: | :----: | :----: |
| noise-free    | 0.924 | 0.900 | 0.894 | 0.881 | 0.875 |
| 0.05          | 0.867 | 0.894 | 0.889 | 0.869 | 0.868 |
| 0.1           | 0.748 | 0.867 | 0.872 | 0.856 | 0.870 |
| 0.15          | 0.619 | 0.696 | 0.832 | 0.854 | 0.851 |
| [0.025, 0.2]  | 0.729 | 0.744 | 0.849 | 0.862 | 0.867 |

#### Learn2Synth: Results for $\sigma = 0$, $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Pre-set | sigma = 0 | low = 0.5/middle = 0.5/high = 0.5 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | -0.0258   | 0.4473/0.5577/0.5587    | 0.910 |
| **model2**    | -0.0205   | 0.4358/0.5567/0.5583    | 0.924 |
| model3        | -0.0204   | 0.4358/0.5567/0.5537    | 0.902 |
| model4        | -0.0201   | 0.4381/0.5566/0.5535    | 0.919 |
| model5        | -0.0045   | 0.5537/0.5578/0.5694    | 0.918 |

#### Learn2Synth: Results for $\sigma = 0.05$, $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Pre-set | sigma = 0.05 | low = 0.5/middle = 0.5/high = 0.5 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.0492    | 0.3438/0.4477/0.7136    | 0.8878    |
| **model2**    | 0.0503    | 0.3198/0.4310/0.7159    | 0.8992    |
| model3        | 0.0488    | 0.3153/0.4288/0.7167    | 0.8977    |
| model4        | 0.0481    | 0.3150/0.4285/0.7166    | 0.8933    |
| model5        | 0.0479    | 0.3148/0.4285/0.7166    | 0.8916    |

#### Learn2Synth: Results for $\sigma = 0.1$, $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Pre-set | sigma = 0.1 | low = 0.5/middle = 0.5/high = 0.5 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.0854    | 0.5856/0.7539/0.9640    | 0.8621    |
| model2        | 0.0853    | 0.5857/0.7541/0.9640    | 0.8624    |
| model3        | 0.0834    | 0.5863/0.7577/0.9661    | 0.8635    |
| model4        | 0.0857    | 0.5870/0.7590/0.9669    | 0.8525    |
| **model5**    | 0.0840    | 0.5887/0.7618/0.9680    | 0.8643    |

#### Learn2Synth: Results for $\sigma = 0.15$, $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Pre-set | sigma = 0.15 | low = 0.5/middle = 0.5/high = 0.5 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.1474    | 0.4012/0.4708/0.6444    | 0.856 |
| **model2**    | 0.1460    | 0.3873/0.4611/0.6498    | 0.858 |
| model3        | 0.1507    | 0.3674/0.4476/0.6594    | 0.847 |
| model4        | 0.1471    | 0.3653/0.4470/0.6598    | 0.856 |
| model5        | 0.1451    | 0.3552/0.4390/0.6612    | 0.854 |

#### Learn2Synth: Results for $\sigma = [0.025, 0.2]$ (regress a single sigma), $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Pre-set | sigma = [0.025, 0.2] | low = 0.5/middle = 0.5/high = 0.5 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.1337    | 0.4221/0.5292/0.8012    | 0.8638    |
| **model2**    | 0.1342    | 0.4064/0.5121/0.7991    | 0.8694    |
| model3        | 0.1342    | 0.4064/0.5120/0.7991    | 0.8618    |
| model4        | 0.1339    | 0.3973/0.5059/0.7986    | 0.8591    |
| model5        | 0.1325    | 0.3932/0.5024/0.7968    | 0.8479    |

#### Learn2Synth: Results for $\sigma = [0.025, 0.2]$ (regress ranges of sigma), $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Pre-set | sigma = [0.025, 0.2] | low = 0.5/middle = 0.5/high = 0.5 | Dice |
| :----: | :----: | :----: | :----: |
| **model1**    | [0.0496, 0.1996]  | 0.3323/0.5632/0.5511    | 0.824 |
| model2        | [0.0497, 0.1997]  | 0.3323/0.5632/0.5511    | 0.807 |
| model3        | [0.0474, 0.1983]  | 0.3317/0.5628/0.5515    | 0.806 |
| model4        | [0.0401, 0.1845]  | 0.3145/0.5492/0.5472    | 0.808 |
| model5        | [0.0409, 0.1833]  | 0.3076/0.5449/0.5456    | 0.821 |


#### Learn2Synth: All settings for $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Setting | Preset Sigma | Opmitized Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | Fixed | 0.5/0.5/0.5   | train_free_05.sh              | Done  |
| Learn2Synth   | 0.05          | Fixed | 0.5/0.5/0.5   | train_005_05.sh               | Done  |
| Learn2Synth   | 0.1           | Fixed | 0.5/0.5/0.5   | train_010_05.sh               | Done  |
| Learn2Synth   | 0.15          | Fixed | 0.5/0.5/0.5   | train_015_05.sh               | Done  |
| Learn2Synth   | [0.025, 0.2]  | Fixed | 0.5/0.5/0.5   | train_vary_05.sh              | Done  |
| Learn2Synth   | [0.025, 0.2]  | Range | -/-/-         | train_vary_vary.sh            | Done  |
| Learn2Synth   | [0.025, 0.2]  | Range | 0.5/0.5/0.5   | train_vary_vary_05_bias.sh    | Done  |

#### SynthSeg: All settings for $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| SynthSeg  | 0             | 0.5/0.5/0.5   | train_synthseg_free_05.sh | Done      |
| SynthSeg  | 0.05          | 0.5/0.5/0.5   | train_synthseg_005_05.sh  | Done      |
| SynthSeg  | 0.1           | 0.5/0.5/0.5   | train_synthseg_010_05.sh  | Done      |
| SynthSeg  | 0.15          | 0.5/0.5/0.5   | train_synthseg_015_05.sh  | Done      |
| SynthSeg  | [0.025, 0.2]  | 0.5/0.5/0.5   | train_synthseg_vary_05.sh | Running   |

#### SynthSeg: All settings for $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$ using learned parameters from Learn2Synth

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| SynthSeg  | -0.0205           | 0.4358/0.5567/0.5583  | train_synthseg_free_05_learned.sh         | Done  |
| SynthSeg  | 0.0503            | 0.3198/0.4310/0.7159  | train_synthseg_005_05_learned.sh          | Done  |
| SynthSeg  | 0.0840            | 0.5887/0.7618/0.9680  | train_synthseg_010_05_learned.sh          | Done  |
| SynthSeg  | 0.1460            | 0.3873/0.4611/0.6498  | train_synthseg_015_05_learned.sh          | Done  |
| SynthSeg  | 0.1342            | 0.4064/0.5121/0.7991  | train_synthseg_vary_05_learned.sh         | Done  |
| SynthSeg  | [0.0496, 0.1996]  | 0.3323/0.5632/0.5511  | train_synthseg_vary_vary_05_learned.sh    | Done  |

#### Learn2Synth: All settings for $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Setting | Preset Sigma | Opmitized Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | Fixed | 0.8/0.8/0.8   | train_free_08.sh              | Done  |
| Learn2Synth   | 0.05          | Fixed | 0.8/0.8/0.8   | train_005_08.sh               | Done  |
| Learn2Synth   | 0.1           | Fixed | 0.8/0.8/0.8   | train_010_08.sh               | Done  |
| Learn2Synth   | 0.15          | Fixed | 0.8/0.8/0.8   | train_015_08.sh               | Done  |
| Learn2Synth   | [0.025, 0.2]  | Fixed | 0.8/0.8/0.8   | train_vary_08.sh              | Done  |
| Learn2Synth   | [0.025, 0.2]  | Range | 0.8/0.8/0.8   | train_vary_vary_08_bias.sh    | Done  |

#### Learn2Synth: Results for $\sigma = 0$, $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Pre-set | sigma = 0 | low = 0.8/middle = 0.8/high = 0.8 | Dice |
| :----: | :----: | :----: | :----: |
| **model1**    | -0.0005   | 0.6519/0.6326/0.7212  | 0.858 |
| model2        | -0.0004   | 0.6520/0.6329/0.7214  | 0.856 |
| model3        | 0         | 0.6520/0.6327/0.7232  | 0.855 |
| model4        | 0.0003    | 0.6515/0.6331/0.7239  | 0.855 |
| model5        | 0.0005    | 0.6511/0.6334/0.7246  | 0.849 |

#### Learn2Synth: Results for $\sigma = 0.05$, $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Pre-set | sigma = 0.05 | low = 0.8/middle = 0.8/high = 0.8 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.0630    | 0.6024/0.5534/0.6216  | 0.830 |
| **model2**    | 0.0630    | 0.6026/0.5534/0.6225  | 0.838 |
| model3        | 0.0633    | 0.6028/0.5535/0.6233  | 0.817 |
| model4        | 0.0631    | 0.6025/0.5541/0.6245  | 0.832 |
| model5        | 0.0642    | 0.6024/0.5542/0.6250  | 0.830 |

#### Learn2Synth: Results for $\sigma = 0.1$, $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Pre-set | sigma = 0.1 | low = 0.8/middle = 0.8/high = 0.8 | Dice |
| :----: | :----: | :----: | :----: |
| **model1**    | 0.1184    | 0.5964/0.5385/0.5916  | 0.814 |
| model2        | 0.1166    | 0.5965/0.5391/0.5924  | 0.806 |
| model3        | 0.1169    | 0.5965/0.5391/0.5925  | 0.808 |
| model4        | 0.1180    | 0.5963/0.5393/0.5932  | 0.814 |
| model5        | 0.1161    | 0.5964/0.5396/0.5942  | 0.814 |

#### Learn2Synth: Results for $\sigma = 0.15$, $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Pre-set | sigma = 0.1 | low = 0.8/middle = 0.8/high = 0.8 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.1543    | 0.6299/0.5168/0.5181  | 0.746 |
| model2        | 0.1578    | 0.6300/0.5169/0.5191  | 0.750 |
| model3        | 0.1590    | 0.6302/0.5172/0.5196  | 0.749 |
| **model4**    | 0.1658    | 0.6291/0.5175/0.5238  | 0.753 |
| model5        | 0.1688    | 0.6286/0.5180/0.5259  | 0.752 |

#### Learn2Synth: Results for $\sigma = [0.025, 0.2]$ (regress a single sigma), $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Pre-set | sigma = [0.025, 0.2] | low = 0.8/middle = 0.8/high = 0.8 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.1450    | 0.6296/0.5256/0.5246    | 0.771   |
| model2        | 0.1604    | 0.6245/0.5250/0.5294    | 0.770   |
| model3        | 0.1617    | 0.6227/0.5254/0.5291    | 0.778   |
| **model4**    | 0.1615    | 0.6228/0.5254/0.5291    | 0.781   |
| model5        | 0.1602    | 0.6222/0.5257/0.5291    | 0.769   |

#### Learn2Synth: Results for $\sigma = [0.025, 0.2]$ (regress ranges of sigma), $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Pre-set | sigma = [0.025, 0.2] | low = 0.8/middle = 0.8/high = 0.8 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | [0.1180, 0.2414]  | 0.6024/0.5456/0.5339  | 0.804 |
| model2        | [0.1172, 0.2410]  | 0.6017/0.5459/0.5343  | 0.804 |
| **model3**    | [0.1170, 0.2411]  | 0.6016/0.5459/0.5344  | 0.815 |
| model4        | [0.1194, 0.2352]  | 0.6016/0.5485/0.5393  | 0.801 |
| model5        | [0.1219, 0.2336]  | 0.6010/0.5487/0.5392  | 0.811 |

#### SynthSeg: All settings for $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| SynthSeg  | 0             | 0.8/0.8/0.8   | train_synthseg_free_08.sh | Running   |
| SynthSeg  | 0.05          | 0.8/0.8/0.8   | train_synthseg_005_08.sh  | Done      |
| SynthSeg  | 0.1           | 0.8/0.8/0.8   | train_synthseg_010_08.sh  | Done      |
| SynthSeg  | 0.15          | 0.8/0.8/0.8   | train_synthseg_015_08.sh  | Done      |
| SynthSeg  | [0.025, 0.2]  | 0.8/0.8/0.8   | train_synthseg_vary_08.sh | Done      |

#### SynthSeg: All settings for $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$ using learned parameters from Learn2Synth

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| SynthSeg  | -0.0005           | 0.6519/0.6326/0.7212  | train_synthseg_free_08_learned.sh         | Running   |
| SynthSeg  | 0.0630            | 0.6026/0.5534/0.6225  | train_synthseg_005_08_learned.sh          | Running   |
| SynthSeg  | 0.1184            | 0.5964/0.5385/0.5916  | train_synthseg_010_08_learned.sh          | Done      |
| SynthSeg  | 0.1658            | 0.6291/0.5175/0.5238  | train_synthseg_015_08_learned.sh          | Running   |
| SynthSeg  | 0.1615            | 0.6228/0.5254/0.5291  | train_synthseg_vary_08_learned.sh         | Running   |
| SynthSeg  | [0.1170, 0.2411]  | 0.6016/0.5459/0.5344  | train_synthseg_vary_vary_08_learned.sh    | Done      |

#### Learn2Synth: All settings for $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Setting | Preset Sigma | Opmitized Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | Fixed | 0.2/0.2/0.2   | train_free_02.sh              | Done  |
| Learn2Synth   | 0.05          | Fixed | 0.2/0.2/0.2   | train_005_02.sh               | Done  |
| Learn2Synth   | 0.1           | Fixed | 0.2/0.2/0.2   | train_010_02.sh               | Done  |
| Learn2Synth   | 0.15          | Fixed | 0.2/0.2/0.2   | train_015_02.sh               | Done  |
| Learn2Synth   | [0.025, 0.2]  | Fixed | 0.2/0.2/0.2   | train_vary_02.sh              | Done  |
| Learn2Synth   | [0.025, 0.2]  | Range | 0.2/0.2/0.2   | train_vary_vary_02_bias.sh    | Done  |

#### Learn2Synth: Results for $\sigma = 0$, $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Pre-set | sigma = 0 | low = 0.2/middle = 0.2/high = 0.2 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0         | 0.2648/0.2881/0.2828  | 0.870 |
| model2        | -0.0013   | 0.2627/0.2869/0.2828  | 0.863 |
| model3        | -0.0010   | 0.2625/0.2867/0.2828  | 0.865 |
| model4        | -0.0012   | 0.2625/0.2867/0.2826  | 0.869 |
| **model5**    | -0.0013   | 0.2624/0.2866/0.2825  | 0.871 |

#### Learn2Synth: Results for $\sigma = 0.05$, $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Pre-set | sigma = 0.05 | low = 0.2/middle = 0.2/high = 0.2 | Dice |
| :----: | :----: | :----: | :----: |
| **model1**    | 0.0486    | 0.3092/0.3407/0.3210  | 0.858 |
| model2        | 0.0480    | 0.3084/0.3394/0.3198  | 0.846 |
| model3        | 0.0491    | 0.3038/0.3357/0.3190  | 0.858 |
| model4        | 0.0491    | 0.3037/0.3358/0.3191  | 0.845 |
| model5        | 0.0489    | 0.2986/0.3318/0.3168  | 0.850 |

#### Learn2Synth: Results for $\sigma = 0.1$, $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Pre-set | sigma = 0.1 | low = 0.2/middle = 0.2/high = 0.2 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.0934    | 0.3390/0.3785/0.3786  | 0.826 |
| model2        | 0.0929    | 0.3387/0.3784/0.3782  | 0.824 |
| **model3**    | 0.0927    | 0.3386/0.3781/0.3781  | 0.829 |
| model4        | 0.0921    | 0.3305/0.3717/0.3718  | 0.814 |
| model5        | 0.0921    | 0.3281/0.3709/0.3698  | 0.829 |

#### Learn2Synth: Results for $\sigma = 0.15$, $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Pre-set | sigma = 0.15 | low = 0.2/middle = 0.2/high = 0.2 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.1497    | 0.3370/0.3750/0.3971  | 0.808 |
| model2        | 0.1497    | 0.3309/0.3709/0.3958  | 0.799 |
| model3        | 0.1510    | 0.3300/0.3703/0.3951  | 0.816 |
| model4        | 0.1492    | 0.3270/0.3682/0.3942  | 0.797 |
| **model5**    | 0.1496    | 0.3233/0.3657/0.3925  | 0.819 |

#### Learn2Synth: Results for $\sigma = [0.025, 0.2]$ (regress a single sigma), $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Pre-set | sigma = sigma = [0.025, 0.2] | low = 0.2/middle = 0.2/high = 0.2 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | 0.1275    | 0.3430/0.3842/0.3702  | 0.805 |
| model2        | 0.1249    | 0.3405/0.3825/0.3693  | 0.812 |
| **model3**    | 0.1238    | 0.3401/0.3826/0.3690  | 0.824 |
| model4        | 0.1267    | 0.3366/0.3808/0.3677  | 0.805 |
| model5        | 0.1260    | 0.3369/0.3801/0.3663  | 0.821 |

#### Learn2Synth: Results for $\sigma = [0.025, 0.2]$ (regress ranges of sigma), $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Pre-set | sigma = sigma = [0.025, 0.2] | low = 0.2/middle = 0.2/high = 0.2 | Dice |
| :----: | :----: | :----: | :----: |
| model1        | [-0.1441, 0.2150] | 0.3470/0.3751/0.5272  | 0.845 |
| model2        | [-0.1465, 0.2138] | 0.3398/0.3698/0.5203  | 0.844 |
| **model3**    | [-0.1496, 0.2135] | 0.3382/0.3685/0.5199  | 0.855 |
| model4        | [-0.1511, 0.2123] | 0.3374/0.3683/0.5198  | 0.848 |
| model5        | [-0.1517, 0.2138] | 0.3373/0.3682/0.5196  | 0.845 |

#### SynthSeg: All settings for $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| SynthSeg  | 0             | 0.2/0.2/0.2   | train_synthseg_free_02.sh | Done  |
| SynthSeg  | 0.05          | 0.2/0.2/0.2   | train_synthseg_005_02.sh  | Done  |
| SynthSeg  | 0.1           | 0.2/0.2/0.2   | train_synthseg_010_02.sh  | Done  |
| SynthSeg  | 0.15          | 0.2/0.2/0.2   | train_synthseg_015_02.sh  | Done  |
| SynthSeg  | [0.025, 0.2]  | 0.2/0.2/0.2   | train_synthseg_vary_02.sh | Done  |

#### SynthSeg: All settings for $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$ using learned parameters from Learn2Synth

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| SynthSeg  | -0.0013           | 0.2624/0.2866/0.2825  | train_synthseg_free_02_learned.sh         | Running   |
| SynthSeg  | 0.0486            | 0.3092/0.3407/0.3210  | train_synthseg_005_02_learned.sh          | Done      |
| SynthSeg  | 0.0927            | 0.3386/0.3781/0.3781  | train_synthseg_010_02_learned.sh          | Done      |
| SynthSeg  | 0.1496            | 0.3233/0.3657/0.3925  | train_synthseg_015_02_learned.sh          | Running   |
| SynthSeg  | 0.1238            | 0.3401/0.3826/0.3690  | train_synthseg_vary_02_learned.sh         | Running   |
| SynthSeg  | [-0.1496, 0.2135] | 0.3382/0.3685/0.5199  | train_synthseg_vary_vary_02_learned.sh    | Running   |

#### Learn2Synth: Ablation study for learning rate

| Setting | Preset Sigma | Opmitized Sigma | Preset low/middle/high | File | LR | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0     | Fixed | 0.5/0.5/0.5   | train_free_05.sh          | 0.0001    | Above     |
| Learn2Synth   | 0     | Fixed | 0.5/0.5/0.5   | train_free_05_lr_0002.sh  | 0.0002    | Done      |
| Learn2Synth   | 0     | Fixed | 0.5/0.5/0.5   | train_free_05_lr_0005.sh  | 0.0005    | &check;   |
| Learn2Synth   | 0     | Fixed | 0.5/0.5/0.5   | train_free_05_lr_001.sh   | 0.001     | &check;   |
| Learn2Synth   | 0     | Fixed | 0.5/0.5/0.5   | train_free_05_lr_002.sh   | 0.002     | Done      |
| Learn2Synth   | 0     | Fixed | 0.5/0.5/0.5   | train_free_05_lr_005.sh   | 0.005     | Fail      |
| Learn2Synth   | 0     | Fixed | 0.5/0.5/0.5   | train_free_05_lr_01.sh    | 0.01      | Fail      |
| Learn2Synth   | 0     | Fixed | 0.5/0.5/0.5   | train_free_05_lr_05.sh    | 0.05      | Fail      |


#### Learn2Synth: Non parametric setting for $c_{low} = 0.5$, $c_{middle} = 0.5$, $c_{high} = 0.5$

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | 0.5/0.5/0.5   | train_non_parametric_free_05.sh   | Running   |
| Learn2Synth   | 0.05          | 0.5/0.5/0.5   | train_non_parametric_005_05.sh    | Running   |
| Learn2Synth   | 0.1           | 0.5/0.5/0.5   | train_non_parametric_010_05.sh    | Running   |
| Learn2Synth   | 0.15          | 0.5/0.5/0.5   | train_non_parametric_015_05.sh    | Running   |
| Learn2Synth   | [0.025, 0.2]  | 0.5/0.5/0.5   | train_non_parametric_vary_05.sh   | Running   |

#### Learn2Synth: Non parametric setting for $c_{low} = 0.8$, $c_{middle} = 0.8$, $c_{high} = 0.8$

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | 0.8/0.8/0.8   | train_non_parametric_free_08.sh   | Running   |
| Learn2Synth   | 0.05          | 0.8/0.8/0.8   | train_non_parametric_005_08.sh    | Running   |
| Learn2Synth   | 0.1           | 0.8/0.8/0.8   | train_non_parametric_010_08.sh    | Running   |
| Learn2Synth   | 0.15          | 0.8/0.8/0.8   | train_non_parametric_015_08.sh    | Running   |
| Learn2Synth   | [0.025, 0.2]  | 0.8/0.8/0.8   | train_non_parametric_vary_08.sh   | To run    |

#### Learn2Synth: Non parametric setting for $c_{low} = 0.2$, $c_{middle} = 0.2$, $c_{high} = 0.2$

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | 0.2/0.2/0.2   | train_non_parametric_free_02.sh   | To run    |
| Learn2Synth   | 0.05          | 0.2/0.2/0.2   | train_non_parametric_005_02.sh    | To run    |
| Learn2Synth   | 0.1           | 0.2/0.2/0.2   | train_non_parametric_010_02.sh    | To run    |
| Learn2Synth   | 0.15          | 0.2/0.2/0.2   | train_non_parametric_015_02.sh    | To run    |
| Learn2Synth   | [0.025, 0.2]  | 0.2/0.2/0.2   | train_non_parametric_vary_02.sh   | To run    |