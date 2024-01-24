# Replaces blank.c in the original kilolib repo and uses their makefile

SRCS := $(wildcard *.c)
HEX := $(SRCS:%.c=build/%.hex)

FOLDER = kilolib-master

all: build ${HEX}

build:
	mkdir -p $@

build/%.hex: %.c | build
	cp $< $(FOLDER)/blank.c
	cd $(FOLDER) && make
	mv $(FOLDER)/build/blank.hex $@

clean:
	rm -fR $(FOLDER)/build
	rm -fR build


