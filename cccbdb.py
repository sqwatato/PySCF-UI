import sys
import runner


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Syntax: python cccbdb.py [molecules...]')
        print('Example: python cccbdb.py CH4')
        exit()

    # run the parser
    for mol in range(1, len(sys.argv)):
        runner.run("geom", sys.argv[mol])
