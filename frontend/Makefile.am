# $Id$

bin_PROGRAMS = classias
#man_MANS = classias.1
#EXTRA_DIST = ${man_MANS}

classias_SOURCES = \
	option.h \
	optparse.h \
	tokenize.h \
	util.h \
	train.h \
	multi_rank.cpp \
	multi_select.cpp \
	multi_classify.cpp \
	main.cpp

#classias_CPPFLAGS =

AM_CFLAGS = @CFLAGS@
INCLUDES = @INCLUDES@
AM_LDFLAGS = @LDFLAGS@

#trankin_CFLAGS = -I$(top_builddir)/lib/crf/include
classias_LDADD = $(top_builddir)/lib/libclassias.la
