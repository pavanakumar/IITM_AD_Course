# AutoDiff Course Material

The repository contains the AutoDiff course material: notes/slides, codes, exercises and reading material. The course uses Fortran and Julia programming language for the demo and exercises. Tapenade AD tool is used for the exercises/demo in Fortran. Emzyme and Zygote is used for AD in Julia language.

We assume that the user has access to a (virtual) machine with GNU/Linux preferably Debian based distro with an active internet connection and a web browser. Install all necessary build/compiler toolchains in the GNU/Linux distro something similar to,

```bash
> sudo apt-get install build-essentials gfortran gcc g++ cmake
```

## Tapenade setup

The hands-on demo and exercises need only the web version of the Tapenade at the URL,

http://www-tapenade.inria.fr:8080/tapenade/index.jsp

### Advanced users

If you wish to install Tapenade on your own machine then make sure you download the latest version from the URL

https://tapenade.gitlabpages.inria.fr/tapenade/

Tapenade needs Java runtime environment which can be installed here,

https://www.java.com/en/download/help/linux_x64_install.html

or on Debian/Ubuntu linux you have something equivalent of

```bash
> sudo apt install default-jre
```

## Fortran compiler

To run the Fortran examples and demo one requires installation of Fortran compilers. Linux comes packaged with the gfortran compiler as part of the gcc tool chain.

```bash
> sudo apt-get install gfortran
```

## Julia setup

You can install Julia using the instructions in the website

https://julialang.org/downloads/

Also install Pluto notebook by following the instructions here,

https://plutojl.org/#install

