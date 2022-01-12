# Transfer Learning using Adversarially Robust ImageNet Models

This repository contains the code and models necessary to replicate the results of our paper:

**Do Adversarially Robust ImageNet Models Transfer Better?** <br>
*Hadi Salman\*, Andrew Ilyas\*, Logan Engstrom, Ashish Kapoor, Aleksander Madry* <br>
Paper: https://arxiv.org/abs/2007.08489 <br>
Blog post:  https://www.microsoft.com/en-us/research/blog/adversarial-robustness-as-a-prior-for-better-transfer-learning/ <br>

```bibtex
    @InProceedings{salman2020adversarially,
        title={Do Adversarially Robust ImageNet Models Transfer Better?},
        author={Hadi Salman and Andrew Ilyas and Logan Engstrom and Ashish Kapoor and Aleksander Madry},
        year={2020},
        booktitle={ArXiv preprint arXiv:2007.08489}
    }
```

## Getting started
*Our code relies on the [MadryLab](http://madry-lab.ml/) public [robustness library](https://github.com/MadryLab/robustness), which will be automatically installed when you follow the instructions below.*
1.  Clone our repo: `git clone https://github.com/microsoft/robust-models-transfer.git`

2.  Install dependencies:
    ```
    conda create -n robust-transfer python=3.7
    conda activate robust-transfer
    pip install -r requirements.txt
    ```


## Running transfer learning experiments
*The entry point of our code is [main.py](src/main.py) (see the file for a full description of arguments).*

1- Download one of the pretrained robust ImageNet models, say an L2-robust ResNet-18 with &epsilon; = 3. For a full list of models, see the [section below](#download-our-robust-imagenet-models)!
  ```
  mkdir pretrained-models & 
  wget -O pretrained-models/resnet-18-l2-eps3.ckpt "https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D"
  ```
2- Run the following script (best parameters for each dataset and architecture can be found [here](best_params_per_dataset_per_arch.md))
  ```
  python src/main.py --arch resnet18 \
    --dataset cifar10 \
    --data /tmp \
    --out-dir outdir \
    --exp-name cifar10-transfer-demo \
    --epochs 150 \
    --lr 0.01 \
    --step-lr 30 \
    --batch-size 64 \
    --weight-decay 5e-4 \
    --adv-train 0 \
    --model-path pretrained-models/resnet-18-l2-eps3.ckpt
    --freeze-level -1
  ```
`--freeze-level -> -1: full-network transfer | 4: fixed-feature transfer`

3- That's it!

## Datasets that we use (see [our paper](https://arxiv.org/abs/2007.08489) for citations) 
* aircraft ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/fgvc-aircraft-2013b.tar.gz?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* birds ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/birdsnap.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* caltech101 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/caltech101.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* caltech256 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/caltech256.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* cifar10 **(Automatically downloaded when you run the code)**
* cifar100 **(Automatically downloaded when you run the code)**
* dtd ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/dtd.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* flowers ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/flowers.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* food ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/food.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* pets ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/pets.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* stanford_cars ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/stanford_cars.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* SUN397 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/SUN397.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))

To use any of these datasets in the code:
1. Download (click or use `wget`) and extract the desired dataset somewhere, e.g. 
    ```
    tar -xvf pets.tar
    mkdir /tmp/datasets
    mv pets /tmp/datasets
    ```
2. Add the dataset name and path as arguments , e.g. 
    ```
    python src/main.py --arch resnet18  ... --dataset pets --data /tmp/datasets/pets
    ```

## Architectures
You can choose an architecture to use by simply passing it as arguments to the code e.g. 
```
python src/main.py --arch resnet50 ...
```
The set of possible architectures is:

```python
archs = [resnet18, 
         resnet50, 
         wide_resnet50_2, 
         wide_resnet50_4, 
         densenet,
         mnasnet,
         mobilenet,
         resnext50_32x4d,
         shufflenet,
         vgg16_bn
         ]
```

## Download our robust ImageNet models
*If you find our pretrained models useful, please consider [citing our work](#transfer-learning-using-adversarially-robust-imagenet-models).*

### Standard Accuracy of L2-Robust ImageNet Models 

|Model|&epsilon;=0|&epsilon;=0.01|&epsilon;=0.03|&epsilon;=0.05|&epsilon;=0.1|&epsilon;=0.25|&epsilon;=0.5|&epsilon;=1.0|&epsilon;=3.0|&epsilon;=5.0|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet-18 |[69.79](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)  | [69.90](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.01.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [69.24](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.03.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [69.15](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.05.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [68.77](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.1.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [67.43](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.25.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [65.49](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [62.32](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps1.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [53.12](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [45.59](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
ResNet-50|[75.80](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [75.68](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.01.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [75.76](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.03.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [75.59](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.05.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [74.78](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.1.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [74.14](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.25.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [73.16](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [70.43](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps1.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [62.83](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [56.13](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
Wide-ResNet-50-2|[76.97](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [77.25](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.01.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [77.26](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.03.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [77.17](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.05.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [76.74](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.1.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [76.21](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.25.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [75.11](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [73.41](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps1.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [66.90](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [60.94](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
Wide-ResNet-50-4|[77.91](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) |[78.02](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps0.01.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)|[77.87](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps0.03.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)|[77.77](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps0.05.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)|[77.64](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps0.1.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)|[77.10](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps0.25.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)|[76.52](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)| [75.51](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps1.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [69.67](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)|[65.20](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)


|Model | &epsilon;=0|&epsilon;=3|
|:-----:|:-----:|:-----:|
 DenseNet |[77.37](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/densenet_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [66.98](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/densenet_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
MNASNET|[60.97](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/mnasnet_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [41.83](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/mnasnet_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
MobileNet-v2|[65.26](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/mobilenet_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [50.40](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/mobilenet_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
ResNeXt50_32x4d|[77.38](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnext50_32x4d_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [66.25](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnext50_32x4d_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
ShuffleNet|[64.25](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/shufflenet_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [43.32](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/shufflenet_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
VGG16_bn|[73.66](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/vgg16_bn_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [57.19](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/vgg16_bn_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)

### Standard Accuracy of Linf-Robust ImageNet Models
|Model|&epsilon;=0.5/255|&epsilon;=1/255|&epsilon;=2/255|&epsilon;=4/255|&epsilon;=8/255|
|---|:---:|:---:|:---:|:---:|:---:|
|ResNet-18|[66.13](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [63.46](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps1.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [59.63](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps2.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [52.49](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps4.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [42.11](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps8.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) 
ResNet-50 |[73.73](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [72.05](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps1.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [69.10](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps2.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [63.86](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps4.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [54.53](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps8.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)
Wide-ResNet-50-2 |[75.82](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [74.65](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps1.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [72.35](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps2.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [68.41](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps4.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) | [60.82](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps8.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D)

# Maintainers

* [Hadi Salman](https://twitter.com/hadisalmanX)
* [Andrew Ilyas](https://twitter.com/andrew_ilyas)
* [Logan Engstrom](https://twitter.com/logan_engstrom) 
* [Ashish Kapoor](https://twitter.com/akapoor_av8r) 
* [Aleksander Madry](https://twitter.com/aleks_madry) 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
