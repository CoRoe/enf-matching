#

all:	test binary

# https://pyinstaller.org/en/stable/usage.html#
binary:
	pyinstaller --onefile --exclude pyinstaller --clean hum.py

test:
	pytest-3 test.py

clean:
	rm -r dist build pyvenv.cfg
