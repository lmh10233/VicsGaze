# VicsGaze
VicsGaze: A Gaze Estimation Method Using Self-supervised Contrastive Learning

# Start
For model training, you need to change the dataset path in `config/train`. Take training ETH-XGaze as an example, you can write code in terminal
```
python train/train_vic.py -s config/train/eth.yaml
```
For model fine-tuning, you need to change the dataset path in `config/test`. If you want to test model on `p00` in MPIIFaceGaze dataset, you can implement
```
python test/leave_linear_eval.py -s config/train/mpii.yaml -t config/test/mpii.yaml -p 0
```
When fine-tuning on Gaze360, you can implement
```
python test/linear_eval.py -s config/train/gaze360.yaml -t config/test/gaze360.yaml
```

# Datasets
The datasets in our paper are open access. You can download at the following link. Remember to cite the corresponding literatures. 
1. [ETH-XGaze](https://ait.ethz.ch/xgaze?query=eth)
2. [Gaze360](http://gaze360.csail.mit.edu/)
3. [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation)
4. [Columbia](https://www.cs.columbia.edu/~brian/projects/columbia_gaze.html)

# Preprocessing
1. [Review and Benchmark](https://phi-ai.buaa.edu.cn/Gazehub/#benchmarks)
```python
@misc{cheng2024appearancebasedgazeestimationdeep,
      title={Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark}, 
      author={Yihua Cheng and Haofei Wang and Yiwei Bao and Feng Lu},
      year={2024},
      eprint={2104.12668},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2104.12668}, 
}
```  
2. [GazeTR](https://github.com/yihuacheng/GazeTR)

```python
@InProceedings{cheng2022gazetr,
  title={Gaze Estimation using Transformer},
  author={Yihua Cheng and Feng Lu},
  journal={International Conference on Pattern Recognition (ICPR)},
  year={2022}
}
```
