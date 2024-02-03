#

all:	test binary

# https://pyinstaller.org/en/stable/usage.html#
binary:
	pyinstaller --onefile \
		--exclude PyQt5-sip \
		--exclude altgraph \
		--exclude charset-normalizer \
		--exclude packaging \
		--exclude pycryptodomex \
		--exclude pyee \
		--exclude pyinstaller \
		--exclude psutil \
		--exclude pyinstaller-hooks-contrib \
		--exclude soupsieve \
		--exclude texttable \
		--exclude typing_extensions \
		--clean hum.py

test:
	pytest-3 test.py

clean:
	rm -r dist build pyvenv.cfg
