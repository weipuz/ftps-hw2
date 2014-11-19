import json, sys

if __name__ == "__main__":
	try:
		inF = open(sys.argv[1], "r")
		outF = open(sys.argv[2], "w")
		startMark = int(sys.argv[3])-1
		endMark = int(sys.argv[4])+startMark
	except:
		print "Usage: python %s <csv-file> <outjsonFile> start-mark-index No-of-grades" %(sys.argv[0])
		print "field delimiter in csv-file is tab"
		sys.exit()
	marks = []
	first = 0
	for line in inF:
		items = line.strip().split("\t")
		if first == 0:
			headers = [a.lower() for a in items]
			print headers
			userIdInd = headers.index("userid")
			first=1
			continue
		person = {"userid":items[userIdInd]}
		for i in range(startMark, endMark):
			person[headers[i]] = {"mark":int(items[i])}
		marks.append(person)
	printable = {"marks":marks}
	print >> outF, json.dumps(printable)
	outF.close
