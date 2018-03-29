# bigan_SRL
Testing BIGAN (Adversarial Feature Learning) for State Representation Learning, a loss term is added to take into account the actions

### Learn a state representation

Usage:
```
python3 main.py [-h] [--dataset {mnist,robot_world}]
               [--dataset_path DATASET_PATH] [--gpu_mode GPU_MODE]
               [--save_dir SAVE_DIR] [--result_dir RESULT_DIR]
               [--log_dir LOG_DIR] [--epoch EPOCH] [--batch_size BATCH_SIZE]
               [--lr LR] [--beta1 BETA1] [--beta2 BETA2] [--slope SLOPE]
               [--decay DECAY] [--dropout DROPOUT] [--network_type {FC,CNN}]
               [--z_dim Z_DIM] [--h_dim H_DIM]


Pytorch implementation of BIGAN

optional arguments:
                -h, --help            show this help message and exit
                --dataset {mnist,robot_world}
                                      The name of dataset
                --dataset_path DATASET_PATH
                --gpu_mode GPU_MODE
                --save_dir SAVE_DIR   Directory name to save the model
                --result_dir RESULT_DIR
                                      Directory name to save the generated images
                --log_dir LOG_DIR     Directory name to save training logs
                --epoch EPOCH         The number of epochs to run
                --batch_size BATCH_SIZE
                                      The size of batch
                --lr LR               Learning rate
                --beta1 BETA1         for adam
                --beta2 BETA2         for adam
                --slope SLOPE         for leaky ReLU
                --decay DECAY         for weight decay
                --dropout DROPOUT
                --network_type {FC,CNN}
                                      Type of network (Fully connectec or CNN)
                --z_dim Z_DIM         The dimension of latent space Z
                --h_dim H_DIM         The dimension of the hidden layers in case of a FC
                                      network
```


Example:
```
python3 main.py --network_type FC --dataset robot_world --epoch 50 --z_dim 2
```
