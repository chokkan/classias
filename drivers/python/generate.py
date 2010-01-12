#!/bin/env python

import sys
import os

def dquote(x):
    return x.replace('"', r'\"')

def source(x):
    dname = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.join(dname, x)

class Binary:
    @staticmethod
    def convert(fi, fo):
        fo.write('weights = {\n')
        for line in fi:
            fields = line.strip('\n').split('\t')
            fo.write('    "%s": %s,\n' % (dquote(fields[1]), fields[0]))
        fo.write('}\n')

    @staticmethod
    def convert_inline(fi, fo):
        f = open(source('binary.py'))
        for line in f:
            if line.startswith('weights = {}'):
                Binary.convert(fi, fo)
            else:
                fo.write(line)

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    fi.readline()
    Binary.convert_inline(fi, fo)

