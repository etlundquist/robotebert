class BOError(Exception):
    def __init__(self, message=''):
        self.message = message

class TweetError(Exception):
    def __init__(self, message=''):
        self.message = message