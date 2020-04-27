# InceptionInspiredLSTMforPredNet
Implementing a paper, [Inception inspired LSTM network](https://arxiv.org/abs/1909.05622 ) which extends [PredNet](https://arxiv.org/abs/1605.08104) for next frame prediction and testing it on [Kitti Dataset](http://www.cvlibs.net/datasets/kitti/)

## Initial setup for scc before running

```
module load cuda/10.1
module load python3/3.6.5
module load pytorch/1.3
```

## Data Setup
Please download the the required train, test Kitti preprocessed data from [here](https://figshare.com/articles/KITTI_hkl_files/7985684) and extract that in './kitti_data' directory

## Commands

### To train PredNet
```
python prednet_train.py
```
### To train Inception based LSTM network 
```
python inception_train.py
```

### To test PredNet
```
python prednet_test.py
```

### To test Inception based LSTM network 
```
python inception_test.py
```

## References
Hosseini, M., Maida, A., Hosseini, M. and Raju, G. (2019). Inception-inspired LSTM for Next-frame Video Prediction. [online] arXiv.org. Available at: https://arxiv.org/abs/1909.05622 [Accessed 1 Mar. 2020].

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Re- search, 32(11):1231–1237, 2013.

Lotter, William, Kreiman, Gabriel, David, and Cox. 2017. “Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning.” ArXiv.org. https://arxiv.org/abs/1605.08104.

GitHub. 2020. Coxlab/Prednet. [online] Available at: <https://github.com/coxlab/prednet> [Accessed 7 April 2020].

GitHub. 2020. Matinhosseiny/Inception-Inspired-LSTM-For-Video-Frame-Prediction. [online] Available at: <https://github.com/matinhosseiny/Inception-inspired-LSTM-for-Video-frame-Prediction> [Accessed 7 April 2020].
