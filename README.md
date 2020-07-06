stl\_normalize.py
================

A script to normalize and validate STL files so that they play better with version control systems like git or mercurial.

Some programs (like OpenSCAD) are highly inconsistent about how they write out STL files.  Small changes to the model can result in major changes in the output STL file.  This makes for larger diffs when checking these files into repositories.

There are also problems with some programs outputting STL files that are not properly formed and manifold.

The stl\_normalize.py script is designed to be run from a Makefile or other build script, to normalize STL files, and verify whether the files are properly manifold. With the validation, you can force a build script to fail on a bad STL, and force the developer to tweak the model to fix the issue.  This script can actually visually show you where the manifold problems are.

This script does the following to normalize STL files:
* Reorders the triangle faces in a consistent physical ordering.
* Reorders triangle vertex data in a consistent way.
* Calculates any missing unit face normals.
* Ensures vertex data is ordered counter-clockwise. (Right-hand rule.)
* Rewrites the file in ASCII STL format.
* Writes vertex coordinate data in a consistent compact way.


Usage
-----

```
stl_normalize.py [-h] [-v] [-c] [-g] [-b] [-o OUTFILE] INFILE
```

Positional argument | What it is
:------------------ | :--------------------------------
INFILE              | Filename of STL file to read in.


Optional arguments             | What it does
:----------------------------- | :--------------------
-h, --help                     | Show help message and exit
-v, --verbose                  | Show verbose output.
-c, --check-manifold           | Perform manifold validation of model.
-g, --gui-display              | Show non-manifold edges in GUI. (using OpenGL)
-b, --write-binary             | Use binary STL format for output.
-o OUTFILE, --outfile OUTFILE  | Write normalized STL to file.


Examples
--------

```
stl_normalize.py -o normalized.stl input.stl
```
This will read in the file ```input.stl```, normalize the data, and write it out as an ASCII STL file named ```normalized.stl```

```
stl_normalize.py -b -o normalize.stl input.stl
```
This will read in the file ```input.stl```, normalize the data, and write it out as a binary STL file named ```normalized.stl```

```
stl_normalize.py -c input.stl
```
This will validate the manifoldness of the file ```input.stl``` and print out any problems it finds. It will return with a non-zero return code if any problems were found.

```
stl_normalize.py -c -o normalized.stl input.stl
```
This will read in the file ```input.stl```, validate its manifoldness, and print out any problems it finds.  If no problems are found, it will normalize the data, and write it out as an ASCII STL file named ```normalized.stl```.  It will return with a non-zero return code if any problems were found.

```
stl_normalize.py -g input.stl
```
This will validate the manifoldness of the file ```input.stl```, and, if there are any problems, launch OpenSCAD to display the non-manifold edges. Holes will be ringed in red. Redundant faces will be ringed in purple.


