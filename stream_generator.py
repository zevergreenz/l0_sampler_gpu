from random import randint

def generate_stream(max_index, num_ops):
	bits = [False] * max_index
	with open("data_stream.txt", "w") as f:
		for i in range(num_ops):
			index = randint(0, max_index - 1)
			update = -1 if bits[index] else 1
			bits[index] = not bits[index]
			print(str(index+1) + " " + str(update) + " ", file=f, end="")

generate_stream(100, 100000)