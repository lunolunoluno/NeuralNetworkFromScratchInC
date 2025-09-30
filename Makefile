CFLAGS:=-std=c99 -Wall -Wextra -fsanitize=address -fsanitize=leak

main: main.c array_utils.o
	gcc $^ -o main.o $(CFLAGS)

array_utils.o: array_utils.c array_utils.h
	gcc $(CFLAGS) -c $<

clean:
	rm -f *.o

rebuild: clean main
