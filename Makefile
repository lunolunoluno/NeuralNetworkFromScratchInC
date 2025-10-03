CFLAGS:=-std=c99 -g -Wall -Wextra -fsanitize=address -fsanitize=leak

main: main.c array_utils.o layer.o
	gcc $^ -o main.o $(CFLAGS)

array_utils.o: array_utils.c array_utils.h
	gcc $(CFLAGS) -c $<

layer.o: layer.c layer.h
	gcc $(CFLAGS) -c $<

clean:
	rm -f *.o

rebuild: clean main
