import os
from deep_pnas_py_src.data import data_load, test_data, test_refnet_data

# Importing different types of Networks
from deep_pnas_py_src.models import build_nn_resnet, reference_nn_resnet
# Importing Training functions
from deep_pnas_py_src.models import train_network, train_refnet_network

from tensorflow.python.client import device_lib
# Just Check if GPU is being used or not
print(device_lib.list_local_devices())

ln_ip_data_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\python_code\ln_train_2m_b3k_input.mat'
ln_op_data_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\python_code\ln_train_2m_b3k_output.mat'

ip_data_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\python_code\b3k_input_2m.mat'
op_data_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\python_code\b3k_output_2m.mat'
test_data_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\python_code\b3k_test_input_1m.mat'

# Loading Data
X, y = data_load(ip_data_path, op_data_path)

# Reduce y to 8th order
y = y[:,:45]

print('Data Loaded ... \n')

res_model = reference_nn_resnet()

print('Network Constructed ... \n')
print('Training Network ... \n')

#res_model = train_network(res_model, X, y, num_epoch=40, batch=1000)
res_model = train_refnet_network(res_model, X, y, num_epoch=100, batch=10000)

print('Making Predictions and Saving file')

save_file_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\Model_Results_2019\seq_refnet.mat'
test_refnet_data(res_model, test_data_path, save_file_path)








