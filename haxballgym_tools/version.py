# https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package

__version__ = '1.0.1'

release_notes = {
    '1.0.1': """
    - Add project description
    """,
    '1.0.0': """
    - Added SB3 environment from rlgym (now called SB3SingleInstanceEnv) and fixed some bugs
    - Added working example code for the SB3 environment
    - Plans to use multiprocessing for the SB3 environment (not yet implemented)
    """
}


def get_current_release_notes():
    if __version__ in release_notes:
        return release_notes[__version__]
    return ''


def print_current_release_notes():
    print(f"Version {__version__}")
    print(get_current_release_notes())
    print("")