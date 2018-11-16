class spexxyException(Exception):
    def __init__(self, message=""):
        self.message = message


class spexxyValueTooLowException(spexxyException):
    pass


class spexxyValueTooHighException(spexxyException):
    pass
