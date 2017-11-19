from random import randint

max_index = 100000
num_updates = 1000000

def generate_stream(max_index, num_ops):
	bits = [False] * max_index
	with open("data_stream.txt", "w") as f:
		for i in range(num_ops):
			index = randint(0, max_index - 1)
			update = -1 if bits[index] else 1
			bits[index] = not bits[index]
			print(str(index+1) + " " + str(update) + " ", file=f, end="")

generate_stream(max_index, num_updates)