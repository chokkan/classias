# $Id$

bin_PROGRAMS = classias-tag
#man_MANS = classias.1
#EXTRA_DIST = ${man_MANS}

classias_tag_SOURCES = \
	../contrib/libexecstream/exec-stream.cpp \
	../contrib/libexecstream/exec-stream.h \
	../include/optparse.h \
	../include/tokenize.h \
	../include/util.h \
	option.h \
	binary.cpp \
	multi.cpp \
	main.cpp

AM_CXXFLAGS = @CXXFLAGS@
INCLUDES = @INCLUDES@ -I../contrib -I../include
AM_LDFLAGS = @LDFLAGS@

