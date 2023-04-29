# import yaml
# import os

# params = {
#         "DATA":{
#             "TRAIN_DATA_DIR": "./data/2D/generated_data",
#             "EVAL_DATA_DIR": "./data/2D/evaluation_data",
#             "SETTING": '2D',
#             "ENVIRONMENT": 'fbcampus',
#         },
#         "MODEL":{
#             "MODEL_NAME": "SimpleRNN",
#             "INPUT_SIZE": 19,
#             "OUTPUT_SIZE": 3,
#             "HIDDEN_SIZE": 128,
#             "NUM_LAYERS": 2,
#             "SEQUENCE_LENGTH": 28,
#         },
#         "SOLVER":{
#             "NUM_EPOCHS": 100,
#             "BATCH_SIZE": 100,
#             "LR": 0.001,
#             "LOG_STEP": 500,
#         },
#         "OUTPUT":{
#             "OUTPUT_DIR": "./output",
#             "PLOT": True,
#         },
#     }

# # convert to yaml
# with open('configs/2D.yaml', 'w') as outfile:
#     yaml.dump(params, outfile, default_flow_style=False)

# with open('configs/1D.yaml', 'w') as outfile:
#     yaml.dump(params, outfile, default_flow_style=False)

