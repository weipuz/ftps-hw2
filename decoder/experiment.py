#!/usr/bin/env python
import optparse, sys, os, logging

start = -10
end = -1
step = 1
for i in xrange(start, end, step):
	e = float(i) / 10.0
	output_file = "log/exp_e" + repr(e) + ".log"
	cmd = "python decoderTry.py -n 3 -e " + repr(e) + " > " + output_file
	print cmd
	ret = os.system(cmd)
	if ret:
		break
	score_file = output_file + ".score"
	os.system("python score-decoder.py < " + output_file + " > " + score_file)
