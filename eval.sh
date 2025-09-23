echo "=============eval Balloon1============="
python eval_nvidia.py -s /share/czh/nvidia_megasam/Balloon1/ --expname "Balloon1" --configs arguments/nvidia_rodynrf/Balloon1.py --checkpoint /share/czh/splinegs_0922/both/Balloon1/point_cloud/fine_best
echo "=============eval Balloon2============="
python eval_nvidia.py -s /share/czh/nvidia_megasam/Balloon2/ --expname "Balloon2" --configs arguments/nvidia_rodynrf/Balloon2.py --checkpoint /share/czh/splinegs_0922/both/Balloon2/point_cloud/fine_best
echo "=============eval Jumping=============="
python eval_nvidia.py -s /share/czh/nvidia_megasam/Jumping/ --expname "Jumping" --configs arguments/nvidia_rodynrf/Jumping.py --checkpoint /share/czh/splinegs_0922/both/Jumping/point_cloud/fine_best
echo "============eval Playground============"
python eval_nvidia.py -s /share/czh/nvidia_megasam/Playground/ --expname "Playground" --configs arguments/nvidia_rodynrf/Playground.py --checkpoint /share/czh/splinegs_0922/both/Playground/point_cloud/fine_best
echo "==============eval Skating============="
python eval_nvidia.py -s /share/czh/nvidia_megasam/Skating/ --expname "Skating" --configs arguments/nvidia_rodynrf/Skating.py --checkpoint /share/czh/splinegs_0922/both/Skating/point_cloud/fine_best
echo "===============eval Truck=============="
python eval_nvidia.py -s /share/czh/nvidia_megasam/Truck/ --expname "Truck" --configs arguments/nvidia_rodynrf/Truck.py --checkpoint /share/czh/splinegs_0922/both/Truck/point_cloud/fine_best
echo "==============eval Umbrella============"
python eval_nvidia.py -s /share/czh/nvidia_megasam/Umbrella/ --expname "Umbrella" --configs arguments/nvidia_rodynrf/Umbrella.py --checkpoint /share/czh/splinegs_0922/both/Umbrella/point_cloud/fine_best