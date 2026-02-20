CFLAGS:=-g -Wall -Wextra -fsanitize=address -fsanitize=leak -lm

main: main.c utils.o layer.o
	gcc $^ -o main.o $(CFLAGS)

utils.o: utils.c utils.h
	gcc $(CFLAGS) -c $<

layer.o: layer.c layer.h
	gcc $(CFLAGS) -c $<

clean:
	rm -f *.o

rebuild: clean main
