# $Id: Makefile.am 17 2008-11-06 07:52:40Z naoaki $

bin_PROGRAMS = classias-train
#man_MANS = classias.1
#EXTRA_DIST = ${man_MANS}

classias_train_SOURCES = \
	option.h \
	optparse.h \
	tokenize.h \
	util.h \
	train.h \
	binary.cpp \
	multi.cpp \
	candidate.cpp \
	main.cpp

AM_CXXFLAGS = @CXXFLAGS@
INCLUDES = @INCLUDES@ -I../include
AM_LDFLAGS = @LDFLAGS@

#trankin_CFLAGS = -I$(top_builddir)/lib/crf/include
classias_train_LDADD = $(top_builddir)/lib/libclassias.la
