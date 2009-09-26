# $Id: Makefile.am 17 2008-11-06 07:52:40Z naoaki $

bin_PROGRAMS = classias-train
#man_MANS = classias.1
EXTRA_DIST = ${man_MANS} \
	../contrib/libexecstream/posix/exec-stream-helpers.h \
	../contrib/libexecstream/posix/exec-stream-helpers.cpp \
	../contrib/libexecstream/posix/exec-stream-impl.cpp

classias_train_SOURCES = \
	../contrib/libexecstream/exec-stream.cpp \
	../contrib/libexecstream/exec-stream.h \
	../include/optparse.h \
	../include/tokenize.h \
	../include/util.h \
	option.h \
	train.h \
	binary.cpp \
	multi.cpp \
	candidate.cpp \
	main.cpp

AM_CXXFLAGS = @CXXFLAGS@
INCLUDES = @INCLUDES@ -I../contrib -I../include
AM_LDFLAGS = @LDFLAGS@

