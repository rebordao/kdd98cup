'''
Contains all methods to do data importation in this project.
'''

import yaml
import pandas as pd

class Importer:

    @staticmethod
    def get_cfg():
        '''
        Loads configuration from yaml file.
        '''

        return yaml.load(open('config.yml', 'rb'))

    @staticmethod
    def get_raw_dat(cfg):
        '''
        Loads raw data as a pandas data frame.
        '''

        # TODO: read this directly from the zip file
        return pd.read_csv(
            'data/' + cfg['data_file'], sep = ',',
            error_bad_lines = False, low_memory = False,
            skip_blank_lines = True, na_values = [' '],
            keep_default_na = True, verbose = True)
