
for model in 'RNN' 'LSTM' 'LSTM_ln' 'GRU'
do    
    # 1D case
    for f in ./data/1D/generated_data/*
    do
        echo "Running experiment: $f"
        python train.py --config_file './configs/1D.yaml' DATA.TRAIN_DATA_DIR $f \
        OUTPUT.OUTPUT_DIR './logs/separate_mae/experiments_1D/' MODEL.TYPE $model DATA.SETUP 'separated'
    done
done