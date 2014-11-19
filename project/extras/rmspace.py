## python scripts for removing spaces from Chinese corpus

import sys

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "USAGE:  python %s <inputfile> <outputfile>" %(sys.argv[0])
		exit(0)
	fin = open(sys.argv[1], "r")
	fout = open(sys.argv[2], "w")
	for line in fin:
		line = line.replace(" ", "").strip()
		print >> fout, line
	fin.close
	fout.close
