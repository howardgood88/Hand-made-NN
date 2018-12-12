num = int(input())

save = []
final = []
for p in range(1, num+1):
	a = 0
	for n in range(1, p+1):
		if p % n == 0 and p != n:
			a += n
			save.append(n)
			#print(p, n)
	if a == p:
		final.append(p)
for i in range(len(final)):
	print(final[i])