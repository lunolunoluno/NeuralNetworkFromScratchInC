CFLAGS:=-g -Wall -Wextra -fsanitize=address -fsanitize=leak -lm

main: main.c array_utils.o layer.o dataset.o function.o
	gcc $^ -o main.o $(CFLAGS)

array_utils.o: array_utils.c array_utils.h
	gcc $(CFLAGS) -c $<

layer.o: layer.c layer.h
	gcc $(CFLAGS) -c $<

dataset.o: dataset.c dataset.h
	gcc $(CFLAGS) -c $<

function.o: function.c function.h
	gcc $(CFLAGS) -c $<

clean:
	rm -f *.o

rebuild: clean main
