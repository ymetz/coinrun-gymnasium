UNAME := $(shell uname -s)
OBJDIRR=.build-release
OBJDIRD=.build-debug

# Check for QTDIR environment variable
ifdef QTDIR
QT_BIN=$(QTDIR)/bin
QT_PKG_CONFIG=$(QTDIR)/lib/pkgconfig
endif

ifeq ($(UNAME),Linux)
 PKG =pkg-config
 MOC =moc
 LIBS =-L/usr/lib64 -lm -lGL -lGLU
 INC =-I/usr/include
 SO=.so
endif

ifeq ($(UNAME),Darwin)
# First check if QTDIR is set
ifdef QTDIR
 MOC =$(QT_BIN)/moc
export PKG_CONFIG_PATH=$(QT_PKG_CONFIG)
# Second check for homebrew Qt
else ifneq ("$(wildcard /usr/local/opt/qt5/bin/moc)", "")
 MOC =/usr/local/opt/qt5/bin/moc
export PKG_CONFIG_PATH=/usr/local/opt/qt5/lib/pkgconfig
# Third check if moc is in PATH (could be Anaconda)
else ifneq ($(shell which moc 2>/dev/null),)
 MOC =$(shell which moc)
# Path to pkg-config files often in the same base directory
QT_BASE=$(dir $(dir $(shell which moc)))
export PKG_CONFIG_PATH=$(QT_BASE)lib/pkgconfig
# As a last resort, just use 'moc' and hope it's in PATH
else
$(warning Mac, no Qt found - defaulting to 'moc' in PATH)
 MOC =moc
endif
 PKG =pkg-config
 SO=.dylib
 
# For Mac with Anaconda, fix include path issues by using libc++ explicitly 
# and ensure proper include search order
 CFLAGS_PLATFORM = -isystem $(QTDIR)/include -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -stdlib=libc++
endif

INC += `$(PKG) --cflags Qt5Widgets`
LIBS += -lstdc++ `$(PKG) --libs Qt5Widgets`
CC=gcc
LINK=gcc
AR=ar r
AR_OUT=
LINK_OUT= -o
MINUS_O = -o
CFLAGS = -std=c++11 -Wall -Wno-unused-variable -Wno-unused-function -Wno-deprecated-register -fPIC -g -O3 -march=native $(CFLAGS_PLATFORM) $(INC)
CFLAGSD = -std=c++11 -Wall -Wno-unused-variable -Wno-unused-function -Wno-deprecated-register -fPIC -g -DDEBUG $(CFLAGS_PLATFORM) $(INC)
SHARED = -shared
DEPENDS = -MMD -MF $@.dep
EVERY_BIN=$(OBJDIRR)/coinrun_cpp$(SO) $(OBJDIRD)/coinrun_cpp_d$(SO)
SRC = \
 coinrun.cpp
SRC_R = $(patsubst %.cpp, $(OBJDIRR)/%.o, $(SRC))
SRC_D = $(patsubst %.cpp, $(OBJDIRD)/%.o, $(SRC))
EVERY_OBJ_R = $(SRC_R)
EVERY_OBJ_D = $(SRC_D)
DEP = $(patsubst %.o,%.o.dep, $(EVERY_OBJ_R) $(EVERY_OBJ_D))
all: dirs $(EVERY_BIN)
$(OBJDIRR)/coinrun.o: .generated/coinrun.moc
.generated/coinrun.moc: coinrun.cpp
	$(MOC) -o $@ $<
$(OBJDIRR)/coinrun_cpp$(SO): $(SRC_R)
	$(LINK) $(SHARED) $(LINK_OUT) $@ $^ $(LIBS)
$(OBJDIRD)/coinrun_cpp_d$(SO): $(SRC_D)
	$(LINK) $(SHARED) $(LINK_OUT) $@ $^ $(LIBS)
$(OBJDIRR)/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< $(MINUS_O)$@ $(DEPENDS)
$(OBJDIRD)/%.o: %.cpp
	$(CC) $(CFLAGSD) -c $< $(MINUS_O)$@ $(DEPENDS)
.PHONY: depends clean dirs
clean:
	$(RM) $(EVERY_BIN) $(EVERY_OBJ_R) $(EVERY_OBJ_D) .generated/*.moc *.ilk *.pdb $(DEP)
	rm -rf .generated
	rm -rf $(OBJDIRD)
	rm -rf $(OBJDIRR)
depends:
	cat $(DEP) > Makefile.dep
.generated:
	mkdir -p .generated
$(OBJDIRR):
	mkdir -p $@
$(OBJDIRD):
	mkdir -p $@
dirs: .generated $(OBJDIRR) $(OBJDIRD)
-include Makefile.dep