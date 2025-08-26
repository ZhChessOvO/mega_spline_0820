# reconstruct on NVIDIA dataset

## setup
follow [splineGS](https://github.com/KAIST-VICLab/SplineGS)

## train
run
```
python train_0822.py -s /share/czh/nvidia_rodynrf/Balloon2/ --expname "Balloon2" --configs arguments/nvidia_rodynrf/Balloon2.py
```

# reconstruct on stereo 4d dataset

## SoM部分

由于该场景不自带mask，需要使用shape of motion的preprocess生成一个

https://github.com/vye16/shape-of-motion/blob/main/preproc/README.md

## megasam部分

`（mega_sam) bash mono_depth_scripts/run_mono-depth_demo.sh `（有时需要魔法上网在线验证）

`(sam2) bash tools/evaluate_demo.sh`

`(sam2) bash tools/evaluate_demo.sh` 

得到：

`/home/czh/code/mega-sam/outputs_cvd/ski_sgd_cvd_hr.npz`

## SplineGS部分

**(和train_stereo.py和config有关的修改已在代码中完成，请直接运行preprocess和train)**

新建train_stereo.py（复制自train_0820.py）

注释掉line364-line443 cam update部分的替换

替换line119的npz路径

检查mask是否为灰度图像：

`python -c "from PIL import Image; img = Image.open('/share/czh/stereo_0815/motion_masks/000.png'); print('RGB' if img.mode == 'RGB' else '灰度' if img.mode in ['L', '1', 'I', 'F'] else img.mode)”`

新建`/home/czh/code/SplineGS/arguments/nvidia_rodynrf/ski.py` （复制自Balloon2.py）

将SoM的mask整理好格式替换进去：/home/czh/code/SplineGS_old/test/organize_mask_0818.py

运行SplineGS的preprocess

训练：`python train_stereo.py -s /share/czh/stereo_0815/ --expname "ski" --configs arguments/nvidia_rodynrf/ski.py`