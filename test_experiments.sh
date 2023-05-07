
echo "Testing experiment on 1D data"
python test.py --config_file './configs/1D.yaml' \
OUTPUT.OUTPUT_DIR './logs/separate/experiments_1D/' DATA.SETUP 'separated'

echo "Testing experiment on 2D data"
python test.py --config_file './configs/2D.yaml' \
OUTPUT.OUTPUT_DIR './logs/separate/experiments_2D/' DATA.SETUP 'separated'