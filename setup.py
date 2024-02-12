# for cxfreeze

# https://cx-freeze.readthedocs.io/en/stable/index.html

import sys
from cx_Freeze import setup, Executable

binincludes = ['libcrypt-eb21b399.so.2']
my_bin_path = './lib/python3.10/site-packages/cx_Freeze/bases/lib/'

#binpaths = ['./lib/python3.10/site-packages/cx_Freeze/bases/lib']

includefiles = [(my_bin_path+'libcrypt-eb21b399.so.2',
                 'libcrypt-eb21b399.so.2')
                ]

build_exe_options = {
    #'bin_includes':  binincludes,
    #'bin_path_includes': binpaths,
    #'include_files': includefiles,
    'silent': 0
}

# base="Win32GUI" should be used only for Windows GUI app
#base = "Win32GUI" if sys.platform == "win32" else None

setup(
    name = "hum",
    version = "0.0",
    description="ENF matching",
    options = {"build_exe": build_exe_options},
    executables = [Executable("hum.py")]
)
