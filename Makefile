TARGET := test1

SOURCES := main.c ./src/*.c
CFLAGS := -I./include -fopenmp

$(TARGET): $(SOURCES)
	@$(CC) $(CFLAGS) -g -o $@ $(SOURCES) -lm -fopenmp

clean:
	rm -f *.o $(TARGET)
