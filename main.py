__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import argparse
import os

import ista, admm, beamforming_attack
from utills import FIGURES_PATH, MATRICES_PATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('-b', '--beamforming', action='store_true')
    parser.add_argument('-i', '--ista', action='store_true')
    parser.add_argument('-a', '--admm', action='store_true')
    args = parser.parse_args()

    # Creating needed paths
    if not os.path.exists(FIGURES_PATH):
        os.mkdir(FIGURES_PATH)

    if not os.path.exists(MATRICES_PATH):
        os.mkdir(MATRICES_PATH)

    if args.beamforming:
        beamforming_attack.execute()
    if args.ista:
        ista.execute()
    if args.admm:
        admm.execute()



