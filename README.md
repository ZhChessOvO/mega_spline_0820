# notion link

https://www.notion.so/MegaSAM-SplineGS-25d8b22ebd8d80369b88f5e26f3d7ac3?source=copy_link

## setup
follow [splineGS](https://github.com/KAIST-VICLab/SplineGS)

## eval stereo example

```
python eval_stereo_0922.py -s /share/czh/stereo4d_moge/pig/ --expname "pig" --configs /home/czh/code/SplineGS/arguments/nvidia_rodynrf/pig.py --checkpoint /share/czh/splinegs_0917/pig/point_cloud/iteration_25000
```