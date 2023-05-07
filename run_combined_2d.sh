for model in 'RNN' 'LSTM' 'LSTM_ln' 'GRU'
do
    # 1D case
    echo "Running experiment on 2D data"
    python train.py --config_file './configs/2D.yaml' \
    OUTPUT.OUTPUT_DIR './logs/appended_mae/experiments_2D/' MODEL.TYPE $model DATA.SETUP 'appended'
done