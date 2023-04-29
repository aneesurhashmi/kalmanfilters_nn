# loop thorugh all the experiments and run them

# loop through  files in dir

# 1D case
for f in ./data/1D/generated_data/*
do
    echo "Running experiment: $f"
    python train.py --config_file './configs/1D.yaml' DATA.TRAIN_DATA_DIR $f \
    OUTPUT.OUTPUT_DIR './logs/experiments_1D/'
done

# 2D case
for f in ./data/2D/generated_data/*
do
    echo "Running experiment: $f"
    python train.py --config_file './configs/2D.yaml' DATA.TRAIN_DATA_DIR $f \
    OUTPUT.OUTPUT_DIR './logs/experiments_2D/'
done