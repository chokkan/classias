# $Id$

noinst_PROGRAMS = \
	classias-train-binary-online \
	classias-train-binary-batch \
	classias-tag-binary

classias_train_binary_online_SOURCES = \
	strsplit.h \
	train_binary_online.cpp

classias_train_binary_batch_SOURCES = \
	strsplit.h \
	train_binary_batch.cpp

classias_tag_binary_SOURCES = \
	strsplit.h \
	tag_binary.cpp

AM_CXXFLAGS = @CXXFLAGS@
INCLUDES = @INCLUDES@ -I../include
AM_LDFLAGS = @LDFLAGS@

