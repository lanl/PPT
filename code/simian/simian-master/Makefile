CURRENT_DIR = $(notdir $(shell pwd))

tbz2: clean
	cd ..; tar -jcvf $(CURRENT_DIR).tbz2 $(CURRENT_DIR)

clean:
	find . -iname "*.pyc" -exec rm {} \;
	find . -iname "._*" -exec rm {} \;
