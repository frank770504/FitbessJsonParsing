import os
import sys

dirr = '../tr_dir'

cmd = 'echo $(ls {})'.format(dirr)

tmp = os.popen(cmd)
tmp = tmp.read().strip().split(" ")

files = []
for f in tmp:
	loc = dirr+ '/' + f
	files.append(loc)

files_str = ' '.join(str(f) for f in files)

cmd = 'python finess_jsp.py {}'.format(files_str)

os.system(cmd)
