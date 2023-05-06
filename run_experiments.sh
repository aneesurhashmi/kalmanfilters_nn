# loop thorugh all the experiments and run them

# loop through  files in dir

# # 1D case
# for f in ./data/1D/generated_data/*
# do
#     echo "Running experiment: $f"
#     python train.py --config_file './configs/1D.yaml' DATA.TRAIN_DATA_DIR $f \
#     OUTPUT.OUTPUT_DIR './logs/experiments_1D/'
# done

# # 2D case
# for f in ./data/2D/generated_data/*
# do
#     echo "Running experiment: $f"
#     python train.py --config_file './configs/2D.yaml' DATA.TRAIN_DATA_DIR $f \
#     OUTPUT.OUTPUT_DIR './logs/experiments_2D/'
# done

# 2D case
echo "Running experiment on 2D data"
python train.py --config_file './configs/2D.yaml' \
OUTPUT.OUTPUT_DIR './logs/experiments_2D/' MODEL.TYPE 'GRU'

# # 1D case
# echo "Running experiment on 1D data"
# python train.py --config_file './configs/1D.yaml' \
# OUTPUT.OUTPUT_DIR './logs/experiments_1D/'


# experiment_2D_ex3:
# equal loss for all output channels
# theta in rad (-1, 1)
# 

# experiment_2D_ex4:
# 