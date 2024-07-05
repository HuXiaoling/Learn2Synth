Experiemnt1:

Get sigma value: model['state_dict']['network.synthnet.sigma']

Note 'vary' means 'Uniformed sampling from [0.025, 0.2]'.

    training on noisy images and test on noisy images as well
        0.05 -> 12
        0.1  -> 5
        0.15 -> 13
        vary -> 15

    training on noise-free images, and test on noisy images
        005_1_1, 11570, 0.6639
        010_1_1, 11030, 0.6400
        015_1_1,  6150, 0.5203
        vary_1_1, 12170, 0.6532

To do: (mean + std for 5 samples)

        train: noise free, test: free, 0.05, 0.1 0.15, vary
    learn2synth: 
        sigma = 0.05 test: free, 0.05, 0.1 0.15,vary
        sigma = 0.1  test: free, 0.1, 0.1 0.15, vary
        sigma = 0.15  test: free, 0.1, 0.1 0.15, vary
        sigma = vary test: free, 0.1, 0.1 0.15, vary

        real real data (OASIS)

        interpolation layer (pytorch)
        Uniformed sampled from same dis. indepdently, upsampling to image size
        goal: the distribution range of bias field
        two learnable weight: the min and max
        [W_min, W_max]

Experimental results on different settings:

(Low bound?) Row is different models, column is different level noise images

| Noise-level | noise-free (SynthSeg) | noise-free (LearnableSynthSeg) | 0.05 | 0.1 | 0.15 | vary | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| noise-free    | 0.9127    | 0.9231    | 0.8995    | 0.8741    | 0.8602    | 0.8661    |
| 0.05          | 0.3100    | 0.3459    | 0.8866    | 0.8730    | 0.8551    | 0.8637    |
| 0.1           | 0.2092    | 0.1717    | 0.7436    | 0.8584    | 0.8540    | 0.8506    |
| 0.15          | 0.1423    | 0.0918    | 0.4917    | 0.7994    | 0.8396    | 0.8332    |
| vary          | 0.2045    | 0.1505    | 0.6353    | 0.8310    | 0.8336    | 0.8386    |

| Noise-level | noise-free |
| :----: | :----: |
| model1    | 0.7212/0.7129 |
| model2    | 0.7280/0.7285 |
| model3    | 0.7174/0.7220 |
| model4    | 0.7019/0.7202 |
| model5    | 0.7301/0.7247 |
| mean      | 0.7197/0.7217 |

Fine-scale sigma results:

| Pre-set sigma | 0.05 | 0.09 | 0.095 | 0.1 | 0.105 | 0.11 | 0.15 | vary |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| model1    | 0.0411    | 0.0829    | 0.0930    | 0.0958    | 0.0997    | 0.1069    | 0.1446    | 0.1328    |
| model2    | 0.0413    | 0.0816    | 0.0925    | 0.0993    | 0.0990    | 0.1040    | 0.1443    | 0.1377    |
| model3    | 0.0402    | 0.0849    | 0.0938    | 0.0981    | 0.0975    | 0.1076    | 0.1457    | 0.1320    |
| model4    | 0.0411    | 0.0858    | 0.0918    | 0.0971    | 0.1010    | 0.1056    | 0.1466    | 0.1318    |
| model5    | 0.0416    | 0.0851    | 0.0934    | 0.0977    | 0.0993    | 0.1067    | 0.1483    | 0.1376    |
| mean      | 0.0411    | 0.0841    | 0.0929    | 0.0976    | 0.0993    | 0.1062    | 0.1459    | 0.1344    |

Fine-scale sigma results:

| Pre-set sigma | 0.095 | 0.096 | 0.097 | 0.098 | 0.099 | 0.1 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| model1    | 0.0934/0.8507 | 0.0900/0.8532 | 0.0931/0.8694 | 0.0942/0.8755 | 0.0935/0.8745 | 0.0958/0.8502 |
| model2    | 0.0938/0.8607 | 0.0891/0.8689 | 0.0939/0.8533 | 0.0946/0.8666 | 0.0946/0.8521 | 0.0993/0.8560 |
| model3    | 0.0951/0.8767 | 0.0882/0.8744 | 0.0909/0.8600 | 0.0944/0.8629 | 0.0942/0.8528 | 0.0981/0.8555 |
| model4    | 0.0945/0.8584 | 0.0899/0.8567 | 0.0913/0.8604 | 0.0948/0.8673 | 0.0941/0.8616 | 0.0971/0.8677 |
| model5    | 0.0933/0.8590 | 0.0897/0.8697 | 0.0929/0.8620 | 0.0965/0.8705 | 0.0983/0.8741 | 0.0977/0.8630 |
| mean      | 0.0940/0.8611 | 0.0894/0.8646 | 0.0924/0.8610 | 0.0949/0.8686 | 0.0949/0.8630 | 0.0976/0.8584 |

Experiment2: synthetic real images = noise_free_image * bias filed + noise

Bias field = ((2 * 2 -> upsampling to 256 * 256) ** (eps_a * a) * ((4 * 4 -> upsampling to 256 * 256) 
** (eps_b * b)) * ((8 * 8 -> upsampling to 256 * 256) ** (eps_c * c))

| Pre-set | sigma = 0.05 | low = 1 | middle = 0 | high = 0 | Dice |
| :----: | :----: | :----: | :----: | :----: | :----: |
| model1    | 0.0508    | 0.1977    | 0.2408    | 0.5661    | 0.8406    |
| model2    | 0.0547    | 0.1615    | 0.2161    | 0.5581    | 0.8514    |
| model3    | 0.0538    | 0.1392    | 0.2050    | 0.5529    | 0.8388    |
| model4    | 0.0535    | 0.1377    | 0.2022    | 0.5514    | 0.8548    |
| model5    | 0.0560    | 0.1334    | 0.1994    | 0.5519    | 0.8515    |

| Pre-set | sigma = 0.10 | low = 1 | middle = 0 | high = 0 | Dice |
| :----: | :----: | :----: | :----: | :----: | :----: |
| model1    | 0.0917    | 0.1266    | 0.1835    | 0.5389    | 0.8362    |
| model2    | 0.0918    | 0.1220    | 0.1832    | 0.5391    | 0.8356    |
| model3    | 0.0920    | 0.1214    | 0.1833    | 0.5386    | 0.8300    |
| model4    | 0.0961    | 0.1083    | 0.1750    | 0.5321    | 0.8290    |
| model5    | 0.0981    | 0.0882    | 0.1581    | 0.5201    | 0.8180    |

| Pre-set | sigma = 0.15 | low = 1 | middle = 0 | high = 0 | Dice |
| :----: | :----: | :----: | :----: | :----: | :----: |
| model1    | 0.1336    | 0.1166    | 0.1822    | 0.5207    | 0.8109    |
| model2    | 0.1361    | 0.1097    | 0.1794    | 0.5168    | 0.8096    |
| model3    | 0.1370    | 0.1040    | 0.1750    | 0.5163    | 0.8240    |
| model4    | 0.1389    | 0.0992    | 0.1721    | 0.5150    | 0.8276    |
| model5    | 0.1402    | 0.0923    | 0.1662    | 0.5090    | 0.8254    |

Check if sigma below the pre-set value

| Model | Sigma | Sigma + 0.2 |
| :----: | :----: | :----: |
| model1    | -0.1055   | 0.0945    |
| model2    | -0.1034   | 0.0966    |
| model3    | -0.1036   | 0.0964    |
| model4    | -0.1046   | 0.0954    |
| model5    | -0.1043   | 0.0957    |
| mean      | -0.1043   | 0.0957    |

| Pre-set | sigma = 0.05 | low = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0498    | 0.4947    | 
| model2    | 0.0506    | 0.4945    |
| model3    | 0.0511    | 0.4922    |
| model4    | 0.0480    | 0.4901    |
| model5    | 0.0489    | 0.4890    |

| Pre-set | sigma = 0.05 | middle = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0490    | 0.5855    | 
| model2    | 0.0486    | 0.5853    |
| model3    | 0.0485    | 0.5853    |
| model4    | 0.0482    | 0.5852    |
| model5    | 0.0480    | 0.5854    |

| Pre-set | sigma = 0.05 | high = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0501    | 0.5742    | 
| model2    | 0.0492    | 0.5738    |
| model3    | 0.0506    | 0.5744    |
| model4    | 0.0511    | 0.5745    |
| model5    | 0.0503    | 0.5740    |

| Pre-set | sigma = 0.1 | low = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0946    | 0.5996    | 
| model2    | 0.0972    | 0.5989    |
| model3    | 0.0962    | 0.5981    |
| model4    | 0.0984    | 0.5965    |
| model5    | 0.0985    | 0.5964    |

| Pre-set | sigma = 0.1 | middle = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.1016    | 0.5767    | 
| model2    | 0.1044    | 0.5773    |
| model3    | 0.1031    | 0.5767    |
| model4    | 0.1036    | 0.5762    |
| model5    | 0.1042    | 0.5761    |

| Pre-set | sigma = 0.1 | high = 0.5 |
| :----: | :----: | :----: |
| model1    | 0.0992    | 0.5991    | 
| model2    | 0.0994    | 0.5995    |
| model3    | 0.1005    | 0.6005    |
| model4    | 0.0991    | 0.6013    |
| model5    | 0.0994    | 0.6014    |

Sigmoid:

| Pre-set | sigma = 0.05 | low = 0.5 | middle = 0.5 | high = 0.5 | Dice |
| :----: | :----: | :----: | :----: | :----: | :----: |
| model1        | 0.0492    | 0.3438    | 0.4477    | 0.7136    | 0.8878    |
| **model2**    | 0.0503    | 0.3198    | 0.4310    | 0.7159    | 0.8992    |
| model3        | 0.0488    | 0.3153    | 0.4288    | 0.7167    | 0.8977    |
| model4        | 0.0481    | 0.3150    | 0.4285    | 0.7166    | 0.8933    |
| model5        | 0.0479    | 0.3148    | 0.4285    | 0.7166    | 0.8916    |

| Pre-set | sigma = 0.1 | low = 0.5 | middle = 0.5 | high = 0.5 | Dice |
| :----: | :----: | :----: | :----: | :----: | :----: |
| model1        | 0.0854    | 0.5856    | 0.7539    | 0.9640    | 0.8621    |
| model2        | 0.0853    | 0.5857    | 0.7541    | 0.9640    | 0.8624    |
| model3        | 0.0834    | 0.5863    | 0.7577    | 0.9661    | 0.8635    |
| model4        | 0.0857    | 0.5870    | 0.7590    | 0.9669    | 0.8525    |
| **model5**    | 0.0840    | 0.5887    | 0.7618    | 0.9680    | 0.8643    |

| Pre-set | sigma = [0.025, 0.2] | low = 0.5 | middle = 0.5 | high = 0.5 | Dice |
| :----: | :----: | :----: | :----: | :----: | :----: |
| model1        | 0.1337    | 0.4221    | 0.5292    | 0.8012    | 0.8638    |
| **model2**    | 0.1342    | 0.4064    | 0.5121    | 0.7991    | 0.8694    |
| model3        | 0.1342    | 0.4064    | 0.5120    | 0.7991    | 0.8618    |
| model4        | 0.1339    | 0.3973    | 0.5059    | 0.7986    | 0.8591    |
| model5        | 0.1325    | 0.3932    | 0.5024    | 0.7968    | 0.8479    |

| Noise-level | sigma = 0.05 | sigma = 0.1 | sigma = [0.025, 0.2] |
| :----: | :----: | :----: | :----: |
|   0.05        |    0.8945  |     |     |
|   0.1         |      |   0.8545  |     |
|[0.025, 0.2]   |      |     |   0.8611  |


| Setting | Preset Sigma | Opmitized Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | Fixed | 0.5/0.5/0.5   | train_free_05.sh          | Running   |
| Learn2Synth   | 0.05          | Fixed | 0.5/0.5/0.5   | train_005_05_bias.sh           | Done      |
| Learn2Synth   | 0.1           | Fixed | 0.5/0.5/0.5   | train_010_05_bias.sh           | Done      |
| Learn2Synth   | 0.15          | Fixed | 0.5/0.5/0.5   | train_015_05.sh           | Running   |
| Learn2Synth   | [0.025, 0.2]  | Fixed | 0.5/0.5/0.5   | train_vary_05_bias.sh     | Done      |
| Learn2Synth   | [0.025, 0.2]  | Range | -/-/-         | train_vary_vary.sh        | Running   |
| Learn2Synth   | [0.025, 0.2]  | Range | 0.5/0.5/0.5   | train_vary_vary_05_bias.sh| Running   |

| Setting | Preset Sigma | Opmitized Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | Fixed | 0.8/0.8/0.8   | train_free_08.sh          | Running   |
| Learn2Synth   | 0.05          | Fixed | 0.8/0.8/0.8   |                           | To run    |
| Learn2Synth   | 0.1           | Fixed | 0.8/0.8/0.8   |                           | To run    |
| Learn2Synth   | 0.15          | Fixed | 0.8/0.8/0.8   |                           | To run    |
| Learn2Synth   | [0.025, 0.2]  | Fixed | 0.8/0.8/0.8   |                           | To run    |
| Learn2Synth   | [0.025, 0.2]  | Range | 0.8/0.8/0.8   | train_vary_vary_08_bias.sh| Running   |

| Setting | Preset Sigma | Opmitized Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Learn2Synth   | 0             | Fixed | 0.2/0.2/0.2   | train_free_02.sh          | Running   |
| Learn2Synth   | 0.05          | Fixed | 0.2/0.2/0.2   |                           | To run    |
| Learn2Synth   | 0.1           | Fixed | 0.2/0.2/0.2   |                           | To run    |
| Learn2Synth   | 0.15          | Fixed | 0.2/0.2/0.2   |                           | To run    |
| Learn2Synth   | [0.025, 0.2]  | Fixed | 0.2/0.2/0.2   |                           | To run    |
| Learn2Synth   | [0.025, 0.2]  | Range | 0.2/0.2/0.2   | train_vary_vary_02_bias.sh| Running   |

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| SynthSeg  | 0             | 0.5/0.5/0.5   | train_synthseg_free_05.sh | Running   |
| SynthSeg  | 0.05          | 0.5/0.5/0.5   | train_synthseg_005_05.sh  | Running   |
| SynthSeg  | 0.1           | 0.5/0.5/0.5   | train_synthseg_010_05.sh  | Running   |
| SynthSeg  | 0.15          | 0.5/0.5/0.5   | train_synthseg_015_05.sh  | Running   |
| SynthSeg  | [0.025, 0.2]  | 0.5/0.5/0.5   | train_synthseg_vary_05.sh | Running   |

| Setting | Preset Sigma | Preset low/middle/high | File | States |
| :----: | :----: | :----: | :----: | :----: |
| SynthSeg  | 0.        |                       |  | To run    |
| SynthSeg  | 0.0503    | 0.3198/0.4310/0.7159  | train_synthseg_005_05_learned.sh  | Running   |
| SynthSeg  | 0.0840    | 0.5887/0.7618/0.9680  | train_synthseg_010_05_learned.sh  | Running   |
| SynthSeg  | 0.        |                       |  | To run    |
| SynthSeg  | 0.1342    | 0.4064/0.5121/0.7991  | train_synthseg_vary_05_learned.sh | Runing    |

