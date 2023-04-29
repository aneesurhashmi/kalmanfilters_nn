






for target_dataset in 'Product' 'Real_World' 'Clipart' # 
do
    python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
    OUTPUT_DIR '../logs/uda/'$model'/office-home/Art2'$target_dataset \
    MODEL.PRETRAIN_PATH '../logs/pretrain/deit_small/office-home/Art/transformer_10.pth' \
    DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' \
    DATASETS.ROOT_TRAIN_DIR2 './data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.NAMES "OfficeHome" DATASETS.NAMES2 "OfficeHome" \
    MODEL.Transformer_TYPE $model_type \

done