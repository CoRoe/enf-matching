#

all:	test binary appimage

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
	pytest-3 -v test.py

clean:
	rm -r dist build pyvenv.cfg AppDir

#freeze:
#	ls -l build/exe.linux-x86_64-3.10/hum

#
# Create an app image. Uses first cx_freeze to create an executable that
# includes the Python3 interpreter but no libraries; the libs are in a
# parallel directory tree.
#
# In a second step, appimagetook creates an app image.
#
appimage:
	python3 setup.py build
	cp build/exe.linux-x86_64-3.10/hum AppDir/usr/bin
	cp -r build/exe.linux-x86_64-3.10/lib/* AppDir/usr/bin/lib
	ARCH=x86_64 appimagetool-x86_64.AppImage AppDir
