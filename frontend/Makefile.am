# $Id: Makefile.am 11 2008-05-05 03:32:50Z naoaki $

bin_PROGRAMS = classias
#man_MANS = classias.1
#EXTRA_DIST = ${man_MANS}

classias_SOURCES = \
	option.h \
	optparse.h \
	tokenize.h \
	util.h \
	train.h \
	multiclass.cpp \
	selector.cpp \
	ranker.cpp \
	main.cpp

#classias_CPPFLAGS =

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
AM_LDFLAGS = @LDFLAGS@

#trankin_CFLAGS = -I$(top_builddir)/lib/crf/include
classias_LDADD = $(top_builddir)/lib/libclassias.la
