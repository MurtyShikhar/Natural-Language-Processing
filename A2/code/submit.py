import sys

model_train = sys.argv[1]
f = open(model_train, "r")

l = f.readlines()
l = map(lambda i: i.strip(), l)
f.close()

model_answers = sys.argv[2]
f = open(model_answers, "r")
ans = f.readlines()
ans = map(lambda i: i.strip(), ans)
f.close()

model_save = sys.argv[3]
f = open(model_save, "w")

for i in xrange(len(l)):
	if not l[i]:
		f.write("\n")
	else:	
		a = l[i]
		b = ans[i]
		f.write(a + " " + b + "\n")

f.close()



 


