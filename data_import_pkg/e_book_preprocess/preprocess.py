'''
Module to process books from project gutenberg
'''

class Preprocess:
    '''
    Class to preprocess books from project gutenberg
    '''
    @staticmethod
    def clean_book(raw_book_path):
        '''
        Function to clean character *, - and _ as well as removing header and footer.
        '''
        # Get raw data
        file = open(raw_book_path, "r", encoding="UTF-8")
        raw_content = file.read()
        file.close()

        # Delete header
        i = raw_content.index('***')
        processed_content = raw_content[i + 30:]

        # Delete footer
        i = processed_content.index('END OF THIS PROJECT GUTENBERG EBOOK')
        processed_content = processed_content[:i]

        # Replace special characters
        special_characters = ['*', '-', '_']
        for character in special_characters:
            processed_content = processed_content.replace(character, '')

        return processed_content

    @staticmethod
    def generate_clean_book(processed_book_path, raw_book_path):
        '''
        Function to generate a clean book
        '''
        # Create new file
        file = open(processed_book_path + '.txt', "w+", encoding="UTF-8")
        file.write(Preprocess.clean_book(raw_book_path))
        file.close()
