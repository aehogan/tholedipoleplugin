from setuptools import setup, Extension
import os
import platform

openmm_dir = '@OPENMM_DIR@'
tholedipoleplugin_header_dir = '@THOLEDIPOLEPLUGIN_HEADER_DIR@'
tholedipoleplugin_library_dir = '@THOLEDIPOLEPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_tholedipoleplugin',
                      sources=['TholeDipolePluginWrapper.cpp'],
                      libraries=['OpenMM', 'TholeDipolePlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), tholedipoleplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), tholedipoleplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='tholedipoleplugin',
      version='1.0',
      py_modules=['tholedipoleplugin'],
      ext_modules=[extension],
     )
