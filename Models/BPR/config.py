# dataset name 
dataset = 'new'
# assert dataset in ['ml-1m', 'pinterest-20']

# paths
main_path = '/opt/ml/BPR-pytorch/data/'

train_rating = main_path + '{}_train_ratings.csv'.format(dataset)
test_rating = main_path + '{}_test_ratings.csv'.format(dataset)
test_negative = main_path + '{}_test_negative.json'.format(dataset)

model_path = './models/'
BPR_model_path = model_path + 'NeuMF.pth'
