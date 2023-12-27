import sys


sys.path.append('//src')
from src.logger import logging


def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = sys.exc_info()  # Correct the order of parameters

    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in Python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    # error_message = f"Error occured in Python script name [{0}] line number [{1}] error message [{2}]".format()
    file_name, exc_tb.tb_lineno, str(error)

    return error_message

    # custom message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    # Override the __str__ method to return the error message

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 1
    except Exception as e:
        logging.info("Divide by zero ")
        raise CustomException(e, sys)
