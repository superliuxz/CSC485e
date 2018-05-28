import sys

def pos_to_essay(path_to_file):

	pos = []

	with open(path_to_file, "r") as fin:
		for line in fin.readlines():
			pos.append( line.strip().split(",")[1] )

	with open(path_to_file[: -4], "w") as fout:
		for p in pos:
			fout.write(p)
		fout.write("\n")

	print(path_to_file)		

if __name__ == "__main__":
	pos_to_essay(sys.argv[1])
