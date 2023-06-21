# Event-based Blurry Frame Interpolation under Blind Exposure

**Official implementation** of the following paper:

Event-based Blurry Frame Interpolation under Blind Exposure by Wenming Weng, Yueyi Zhang, Zhiwei Xiong. In CVPR 2023.

## Dataset

Please follow the instructions from directory `generate_dataset` to generate the synthetic dataset.
Our collected real-world dataset RealBlur-DAVIS can be downloaded from this [site](https://rec.ustc.edu.cn/share/421c2a20-0fd3-11ee-bf10-0bc6e486a7d3) with password: 64l7.

## Pretrained model

The pretrained model will be released in this [site](https://rec.ustc.edu.cn/share/7582d0c0-0fd3-11ee-9b7b-233132bcb7d9) with password: uvon.

## Training and Inference

Please check the file `scripts\train_ours.sh` and `scripts\infer_ours.sh` for training and inference. 

## Citation

If you find this work helpful, please consider citing our paper.

```latex
@InProceedings{Weng_2023_CVPR,
    author    = {Weng, Wenming and Zhang, Yueyi and Xiong, Zhiwei},
    title     = {Event-based Blurry Frame Interpolation under Blind Exposure},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
 (CVPR)},
    year      = {2023},
}
```

## Contact

If you have any problem about the released code, please do not hesitate to contact me with email (wmweng@mail.ustc.edu.cn).
