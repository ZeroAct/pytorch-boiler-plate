import os
import argparse
import configparser

def is_str(c):
    def is_int(c):
        try:
            int(c)
            return True
        except:
            return False
    
    def is_float(c):
        try:
            float(c)
            return True
        except:
            return False
    
    if is_int(c) or is_float(c):
        return False
    else:
        return True

class BaseOpt:
    
    """
    This class is an Base Opt class.
    
    """
    
    def __init__(self, config_path="opt/base.ini"):
        
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-n', '--name', type=str, default="base",
                                 help="Model name. The weights files will be saved with this name.")
        self.parser.add_argument('-d', '--weights_dir', type=str,
                                 help="Weights directory.")
        
        self.parser.add_argument('-e', '--epochs', type=int, default=1,
                                 help="training epochs")
        self.parser.add_argument('-lr', '--lr', type=float, default=0.001,
                                 help="learning rate")
        
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help="batch size")
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help="number of threads")
        self.parser.add_argument('--shuffle', type=bool, default=True,
                                 help="dataset shuffle")
        self.parser.add_argument('--drop_last', type=bool, default=True,
                                 help="dataset drop last")
        
        self.load_config(config_path)
        self.set_custom_defaults()
    
    def set_custom_defaults(self):
        name = self.parser.get_default('name')
        
        project_dir = os.path.join("project", name)
        os.makedirs(project_dir, exist_ok=True)
        weights_dir = os.path.join(project_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        log_dir = os.path.join(project_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        
        self.parser.set_defaults(project_dir=project_dir)
        self.parser.set_defaults(weights_dir=weights_dir)
        self.parser.set_defaults(log_dir=log_dir)
    
    def get_args(self):
        opt = self.parser.parse_args()
        
        return opt
    
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                print(''.join(list(filter(lambda x: x!='\n', f.readlines()))))
        except:
            raise FileNotFoundError
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        sections = config.sections()
        for section in sections:
            options = config.options(section)
            for option in options:
                val = config.get(section, option)
                if not is_str(val):
                    val = eval(val)
                else:
                    val = "'" + val + "'"
                eval(f"self.parser.set_defaults({option}={val})")
    