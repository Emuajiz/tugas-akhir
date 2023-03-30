import base64

tmp = []
for i in range(10, 20):
    tmp.append(chr(i))
print(tmp)
print(''.join(tmp).encode("ascii"))