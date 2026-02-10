**IQFormerLite: A Hardware-Efficient Framework for Real-Time Automatic Modulation Classification on Edge NPUs**

## **Preparation**

We conducted experiments on two datasets, namely RML2016.10a, RML2016.10b. The datasets can be downloaded from the [DeepSig](https://www.deepsig.ai/datasets/)(RML2016 series).


Please extract the downloaded compressed file directly into the `./dataset` directory, or change the `args.database_path`.  `args.database_choose`  should keep in [2016.10a, 2016.10b].

Then you just need to 

```python
bash train.sh
```

## **Environment**

These models are implemented in Keras, and the environment setting is:

- Python 3.11
- Pytorch 1.12.0
- pandas
- seaborn
- h5py
- scikit-learn
- matplotlib
- tensorboardX
- tqdm
- timm

## **License**

This code is distributed under an [MIT LICENSE](https://github.com/zjwXDU/AMC-Net/blob/main/LICENSE). Note that our code depends on other libraries and datasets which each have their own respective licenses that must also be followed.
