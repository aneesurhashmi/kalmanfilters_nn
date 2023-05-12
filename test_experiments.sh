
echo "Testing experiment on 2D data"
python test.py --config_file './configs/2D.yaml' \
OUTPUT.OUTPUT_DIR './logs/separate_mae/experiments_2D/' DATA.SETUP 'separated' SOLVER.BATCH_SIZE 28

# echo "Testing experiment on 2D data"
# python test.py --config_file './configs/2D.yaml' \
# OUTPUT.OUTPUT_DIR './logs/separate/experiments_2D/' DATA.SETUP 'separated'