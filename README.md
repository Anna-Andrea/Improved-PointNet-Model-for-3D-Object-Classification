# Training and Evaluation Results

## Txt Files Containing Training and Evaluation Results

- **ModelNet40 Dataset:**  
  - Path: `../data/modelnet40_ply_hdf5_2048`

- **ShapeNetPart Dataset:**  
  - *Note:* To download the dataset, run the following command:  
    ```
    cd part_seg
    sh download_data.sh
    ```

- **Baseline PointNet:**  
  - Training Results: `log/baseline/log_train.txt`  
  - Test Results: `dump/baseline/log_test.txt`

- **PointNet_SE Model:**  
  - Training Results: `log/se-block/log_train.txt`  
  - Test Results: `dump/se-block/log_test.txt`

- **PointNet_SE_BN Model:**  
  - Training Results: `log/bn-layer/log_train.txt`  
  - Test Results: `dump/bn-layer/log_test.txt`

- **PointNet_SE_BN_Deeper Model:**  
  - Training Results: `log/bn-layer/log_train.txt`  
  - Test Results: `dump/bn-layer/log_test.txt`

- **PointNet_SE_BN on ShapeNetPart Dataset:**  
  - Training Results: `part_seg/train_results/logs/log.txt`  
  - Test Results: `part_seg/test_results/log.txt`

## Linux Commands for Training and Testing

### Baseline PointNet
- **Training:** `python train.py`
- **Testing:** `python test.py --visu`

### PointNet_SE Model
- **Training:** `python train.py --model pointnet_cls_se`
- **Testing:** `python test.py --model pointnet_cls_se --visu`

### PointNet_SE_BN Model
- **Training:** `python train.py --model pointnet_cls_se_bn`
- **Testing:** `python test.py --model pointnet_cls_se_bn --visu`

### PointNet_SE_BN_Deeper Model
- **Training:** `python train.py --model pointnet_cls_se_bn_deeper`
- **Testing:** `python test.py --model pointnet_cls_se_bn_deeper --visu`

### Training and Testing on ShapeNetPart Dataset
- **Training:**  
