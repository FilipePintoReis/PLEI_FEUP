'''
Contains LinceFileReader
'''

class LinceFileReader:
    '''
    This class read files from lince dataset when given a prefix.
    The file structure inside the prefix folder is ought to be:
        prefix/
            train.conll
            test.conll
            dev.conll
    '''
    @staticmethod
    def train_data(prefix):
        '''
        Returns training data based on a prefix
        '''
        train_str = f"{prefix}/train.conll"
        train_file = open(train_str, "r", encoding="UTF-8")
        train_string = train_file.read()
        train_file.close()
        return train_string

    @staticmethod
    def test_data(prefix):
        '''
        Returns test data based on a prefix
        '''
        test_str = f"{prefix}/test.conll"
        test_file = open(test_str, "r", encoding="UTF-8")
        test_string = test_file.read()
        test_file.close()
        return test_string

    @staticmethod
    def dev_data(prefix):
        '''
        Returns validation data based on a prefix
        '''
        dev_str = f"{prefix}/dev.conll"
        dev_file = open(dev_str, "r", encoding="UTF-8")
        dev_string = dev_file.read()
        dev_file.close()
        return dev_string