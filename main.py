__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('-b', '--beamforming', action='store_true')
    parser.add_argument('-i', '--ista', action='store_true')
    parser.add_argument('-i', '--admm', action='store_true')
    parser.parse_args()


