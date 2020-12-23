# Efficient Fully Convolution Neural Network for Generating Pixel Wise Robotic Grasps With High Resolution Images

### [[Paper](https://ieeexplore.ieee.org/abstract/document/8961711)]

<img src="doc/demo_grasp.png" style="zoom:50%;" />

This paper presents an efficient neural network model to generate  robotic grasps with high resolution images. The proposed model uses  fully convolution neural network to generate robotic grasps for each  pixel using 400 × 400 high resolution RGB-D images. It first down-sample the images to get features and then up-sample those features to the  original size of the input as well as combines local and global features from different feature maps. Compared to other regression or  classification methods for detecting robotic grasps, our method looks  more like the segmentation methods which solves the problem through  pixel-wise ways. We use Cornell Grasp Dataset to train and evaluate the  model and get high accuracy about 94.42% for image-wise and 91.02% for  object-wise and fast prediction time about 8 ms. We also demonstrate that without training on the multiple objects dataset, our model can  directly output robotic grasps candidates for different objects because  of the pixel wise implementation.

## Prerequisites

- Linux
- Anaconda
- Python (Test with Python 3.6)
- PyTorch (Test with 1.0.0 with CUDA 9.0)

## Quick start

#### Cornell Dataset 

- prepare dataset:

Download Cornell dataset and unzip it.

```bash
python generate_dataset.py \
		--datasets /media/ros/0D1416760D141676/Documents/CORNELL/cornell \
		--output_dir /media/ros/0D1416760D141676/Documents/CORNELL/datasets
```

- train model:
```bash
python train.py \
		--datasets /media/ros/0D1416760D141676/Documents/CORNELL/datasets \
		--bachSize 16 \
		--nepoch 500 \
		--outf out 
```

- evaluate model:
```bash
python evaluate.py \
		--datasets /media/ros/0D1416760D141676/Documents/CORNELL/datasets \
		--model xx.pth
```


#### Georgia Dataset 

- prepare dataset:

Download Georgia dataset and unzip it.

```bash
python multi/generate_multi_dataset.py \
		--datasets /home/ros/Documents/Grasp/grasp_multiObject/rgbd \
		--output_dir /media/ros/0D1416760D141676/Documents/CORNELL/multi_datasets
```

- evaluate model:
```bash
python multi/evaluate_multi.py \
		--datasets /media/ros/0D1416760D141676/Documents/CORNELL/multi_datasets \
		--model xx.pth
```

## License

This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

If you find this code useful for your research, please consider citing the following paper:

	@inproceedings{wang2019efficient,
	  title={Efficient fully convolution neural network for generating pixel wise robotic grasps with high resolution images},
	  author={Wang, Shengfan and Jiang, Xin and Zhao, Jie and Wang, Xiaoman and Zhou, Weiguo and Liu, Yunhui},
	  booktitle={2019 IEEE International Conference on Robotics and Biomimetics (ROBIO)},
	  pages={474--480},
	  year={2019},
	  organization={IEEE}
	}

## Acknowledgments
- [GG-CNN](https://github.com/dougsm/ggcnn)

