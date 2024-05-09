## Txt files which contain the results for training and evaluation periods.

ModelNet40 dataset: ../data/modelnet40_ply_hdf5_2048

ShapeNetPart dataset (Please run command): 
cd part_seg
sh download_data.sh

Train results of baseline PointNet: 
../log/baseline/log_train.txt

Test results of baseline PointNet: 
../dump/baseline/log_test.txt

Train results of PointNet_SE: 
../log/se-block/log_train.txt

Test results of PointNet_SE: 
../dump/se-block/log_test.txt

Train results of PointNet_SE_BN: 
../log/bn-layer/log_train.txt

Test results of PointNet_SE_BN: 
../dump/bn-layer/log_test.txt

Train results of PointNet_SE_BN_Deeper: 
../log/bn-layer/log_train.txt

Test results of PointNet_SE_BN_Deeper: 
../dump/bn-layer/log_test.txt

Train results of PointNet_SE_BN on ShapeNetPart dataset: 
../part_seg/train_results/logs/log.txt

Test results of PointNet_SE_BN on ShapeNetPart dataset: 
../part_seg/test_results/log.txt


## Linux Command for training and testing different models (Training will cost much more time, while testing is soon)

Training baseline PointNet: 
python train.py

Testing baseline PointNet: 
python test.py --visu

Training PointNet_SE model:
python train.py --model pointnet_cls_se

Testinig PointNet_SE model:
python test.py --model pointnet_cls_se --visu

Training PointNet_SE_BN model:
python train.py --model pointnet_cls_se_bn

Testinig PointNet_SE_BN model:
python test.py --model pointnet_cls_se_bn --visu

Training PointNet_SE_BN_Deeper model:
python train.py --model pointnet_cls_se_bn_deeper

Testinig PointNet_SE_BN_Deeper model:
python test.py --model pointnet_cls_se_bn_deeper --visu

Training PointNet_SE_BN on ShapeNetPart dataset:
cd part_seg
sh download_data.sh
python train.py

Training PointNet_SE_BN on ShapeNetPart dataset:
cd part_seg
python test.py
