# CSVSR(IEEE TGRS 2024)
📖[Paper](https://ieeexplore.ieee.org/document/10438488) |🖼️[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10438488)

PyTorch codes for 《Deformable Convolution Alignment and Dynamic Scale-Aware Network for Continuous-Scale Satellite Video Super-Resolution》, IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2024.

Authors: Ning Ni, and Libao Zhang (Beijing Normal University)

## Framework ⚓
![image](https://github.com/chongningni/CSVSR/assets/58589797/15211957-dde1-4daa-9e00-907a4f1cf304)

## Usage 🗝
```
git clone https://github.com/chongningni/CSVSR.git
```
## Requirements 🛒
* pytorch==0.4.0
* cffi==1.15.0
* einops==0.3.2
* h5py==3.1.0
* matplotlib==3.3.4
* numpy==1.19.5
* Pillow==9.2.0
* progressbar33==2.4
* scikit_image==0.17.2
* scipy==1.5.4
* setuptools==58.0.4
* tensorboardX==2.5.1
* torchvision==0.2.2
* tqdm==4.19.9

## Test👇
```
python test_csvsr.py --scales 4.0 --test-set Set001 -mp log/CSVSR/version_0/check_points/x2,3,4/best_model.pt -sp res
```

## Train👇
```
python train.py --model CSVSR --train-set data/VSR_data --test-set data/benchmarks/Set001 --checkpoint checkpoint --verbose -n 200 -b 16 -ss 2,3,4 -p 32
```

* Where "data/VSR_data" can be replaced by your traning datasets path, the structure as follows:

![image](https://github.com/chongningni/CSVSR/assets/58589797/63c18413-0ee2-4c25-9842-51ded5155d09)



## Quantitative and Visual Results 👀
![image](https://github.com/chongningni/CSVSR/assets/58589797/68917550-be25-4b8c-b42e-52eb5f53eb19)

![image](https://github.com/chongningni/CSVSR/assets/58589797/93aebf22-25f0-4598-8158-32a89c2285d2)

## Citation🤝
If you find our work helpful in your research, please consider citing it. Thanks! 🤞
```
@articla{ni2024deformable,
    author={Ni, Ning and Zhang, Libao},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Deformable Convolution Alignment and Dynamic Scale-Aware Network for Continuous-Scale Satellite Video Super-Resolution}, 
    year={2024},
    volume={62},
    number={},
    pages={1-17},
    doi={10.1109/TGRS.2024.3366550}
}
```

```
N. Ni and L. Zhang, "Deformable Convolution Alignment and Dynamic Scale-Aware Network for Continuous-Scale Satellite Video Super-Resolution," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-17, 2024, Art no. 5610017, doi: 10.1109/TGRS.2024.3366550.
```

## Acknowledgement🙏
Our code is built upon [TDAN](https://github.com/YapengTian/TDAN-VSR-CVPR-2020).

## Update◐◓◑◒
Satellite data is continuously updated……………………………………………………………………………………


