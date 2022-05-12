original_file = open("test/part.txt", "r")
f1 = open("test/1/1.txt", "a")
f0 = open("test/0/0.txt", "a")
Lines = original_file.readlines()

for line in Lines:
    if (line[-2] == "0"):
        f0.write(line[:-2]+"\n")
    else:
        f1.write(line[:-2]+"\n")

original_file.close()
f1.close()
f0.close()
