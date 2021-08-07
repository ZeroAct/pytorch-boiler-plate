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
        
        self.parser.add_argument('-e', '--epochs', type=int,
                                 help="training epochs")
        self.parser.add_argument('-lr', '--lr', type=float,
                                 help="learning rate")
        self.parser.add_argument('-b', '--batch_size', type=int,
                                 help="batch size")
        
        self.load_config(config_path)
        self.set_custom_defaults()
    
    def set_custom_defaults(self):
        self.parser.set_defaults(weights_dir=f"weights/{self.parser.get_default('name')}")
    
    def get_args(self):
        return self.parser.parse_args()
    
    def load_config(self, config_path):
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
