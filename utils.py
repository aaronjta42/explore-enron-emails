"""
Class for handling logging tasks
"""

import os
import time
import errno
import functools


def pprint(str=None):
    print("[{}] {}".format("INFO", str))


def timeit(func):
    """
    Timing decorator to time functions

    :param func:    function to time
    :return:        func parameter wrapped in a timing function
    """

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        res = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        pprint('function [{}] finished in {} ms'.format(func.__name__, int(elapsedTime * 1000)))
        return res

    return newfunc


@timeit
def init_directory(directory=None):
    """
    Creates directory if doesn't already exist

    :param directory:   str, directory path
    :return:            str, directory path
    """

    try:
        os.makedirs(directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            pprint("os.makedirs failed on {}".format(directory))
            raise

    return directory