import yaml, argparse
from utils import dotdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help="configuration file *.yml", type=str, required=False, default='./config.yml')
    parser.add_argument('--test', help="test file *.yml", type=str, required=False, default='test')
    args = parser.parse_args()

    with open('./config.yml') as f:
        config = yaml.safe_load(f)
    print(config)
    
    opt = dotdict(vars(config))

    print(opt.test)