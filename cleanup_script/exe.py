import os
import subprocess
import time
import argparse


"""
Given overdensity evolution deltalin0 and redshfit z,
it will generate corresponding Landau damping rate caused by electron cosmic rays
with k range from 0.995 to 5 times omega_p/c
"""


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    time_start = time.time()
    z = args.z
    deltalin0 = args.deltalin0

    timestr = time.strftime("%Y%m%d-%H%M%S")
    PATH = f'./results/redshift{z}_density{deltalin0}_{timestr}/'
    create_directory(PATH)

    if not os.path.exists('source_term_Khaire.txt'):
        # Run get_source.py
        print('executing get_source.py...')
        subprocess.run(['python', 'get_source.py'], check=True)
        print('finished!')

    # Run IGM.py with specified arguments
    print('executing IGM.py')
    subprocess.run([
        'python', 'IGM.py',
        '--deltalin0', str(deltalin0),
        '--savePATH', PATH
    ], check=True)
    print('finished!')

    # Run damp_rate.py with specified arguments
    print('executing damp_rate.py')
    subprocess.run([
        'python', 'damp_rate.py',
        '--deltalin0', str(deltalin0),
        '--z', str(z),
        '--savePATH', PATH,
        '--readPATH', PATH
    ], check=True)
    print(f'All process finished! The whole process takes {(time.time-time_start)/60} minutes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--deltalin0', type=float, help='overdensity evolution, 0 for mean density, > 0 for overdensity, and < 0 for underdensity', default=0)
    parser.add_argument('--z', type=float, help='desired redshift', default=2)

    args = parser.parse_args()
    
    main(args)