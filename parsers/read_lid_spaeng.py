
class LinceFileReader:
    @staticmethod
    def train_data(prefix):
        train_str = f"{prefix}/train.conll"
        train_file = open(train_str, "r", encoding="UTF-8")
        train_string = train_file.read()
        train_file.close()
        return train_string

    @staticmethod
    def test_data(prefix):
        test_str = f"{prefix}/test.conll"
        test_file = open(test_str, "r", encoding="UTF-8")
        test_string = test_file.read()
        test_file.close()
        return test_string

    @staticmethod
    def dev_data(prefix):
        dev_str = f"{prefix}/dev.conll"
        dev_file = open(dev_str, "r", encoding="UTF-8")
        dev_string = dev_file.read()
        dev_file.close()
        return dev_string


print(LinceFileReader.train_data('lid_spaeng'))