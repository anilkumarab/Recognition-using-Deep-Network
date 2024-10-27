# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This file is a part of task_4

import wandb
import integrated_train

# NOTE - we use Weights and Biases (wandb) only for visualizing all the data, and for hyperparameter search strategy (basically randomizing the parameters)

sweep_configuration = {
    "method" : "random"
}

metric = {  # we're only storing test_accuracy in the wandb database along its respective hyperparameters
    "name" : "test_accuracy",
    'goal' : "maximize"
}

sweep_configuration["metric"] = metric

# these are the parameters that we're gone hypertune - task_4
parameters = {
    "conv_layer" : {  # size of filters in the first conv. layer
        "values" : [8, 10, 12]
    },
    "conv_kernel" : {  # kernel size of all the conv. layers
        "values" : [3, 5, 7]
    },
    "pool_kernel" : {  # pool size of 2nd max-pool layer (mathematical constraints are the reason to be more specific)
        "values" : [2, 3, 4]
    },
    "batch_size" : {  # batch size
        "values" : [8, 16, 32, 64]
    },
    "lr" : {  # learning rate
        "values" : [1e-1, 1e-2, 1e-3]
    },
    "epochs" : {  # no of epochs
        "values" : [20, 30, 50, 70, 90]
    },
    "fc" : {  # no. of nodes in fc0 (refer NN_Model for architecture)
        "values" : [30, 50, 100]
    },
    "dropout" : {  # dropout
        "values" : [0.3, 0.5, 0.7]
    }
}

# total possibilities ~ 14500
sweep_configuration["parameters"] = parameters
# extension(1/3) for using external software (for visualization and randomizing the parameters only) in the process of automation
# + we've also used neptune.ai for visualizing the model/ network architecture (refer to report)

# "automated_testing_phase2" is the name of the project in which we're storing the data
sweep_id = wandb.sweep(sweep_configuration, project= "automated_testing_phase2")
wandb.agent(sweep_id, integrated_train.start_process, count= 1)

# so whenever the test_wandb.py is called, it'll outsource the training part to integrated_train.py to calculate the test_accuracy of the current parameters
# the test accuracy along with its hyperparameters data are sent to the wandb database, so that we can visualize the data (and also analyze) in the future

# also this process of training is completely automated, the only thing we've to do is to run the tst.sh (bash script), where it runs 35 models simultaneously for 10 times
# NOTE - the number 35 in the tst.sh is based on the GPU performance of our laptop, it needs to be changed according to the user's GPU capability

# extension(2/3) for training more variations
# Also after analyzing the results of almost 1200 possibilities/ variations (we've attached the images of wandb in the report) we found that the best hyperparameters for the model are (individually)
# conv_layer  - 10       - change(1.2%)
# conv_kernel - 3        - change(0.6%)
# pool_kernel - 2        - change(1.1%)
# batch_size  - 32       - change(4.5%) - more important
# lr          - 1e-2     - change(4.0%) - more important
# epochs      - 30       - change(0.8%)
# fc          - 100      - change(0.8%)
# dropout     - 0.5, 0.3 - change(0.7%)

# higher is the change higher the importance (change % is nothing but the difference of the avg. best and avg. worst result of a dimension)

# the best test accuracy that we've got is 98.254%
# where batch_size - 16, conv_layer - 12, conv_kernel - 5, epochs - 70, dropout - 0.5, fc - 100, lr - 1e-2, pool_kernel - 2
