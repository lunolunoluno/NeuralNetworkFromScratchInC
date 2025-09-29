CFLAGS:=-std=c99 -Wall -Wextra -fsanitize=address -fsanitize=leak

main.o: main.c 
	gcc $^ -o $@ $(CFLAGS)
