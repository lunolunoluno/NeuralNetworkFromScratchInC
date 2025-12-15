CFLAGS:=-g -Wall -Wextra -fsanitize=address -fsanitize=leak -lm

main: main.c
	gcc $^ -o main.o $(CFLAGS)

clean:
	rm -f *.o

rebuild: clean main
