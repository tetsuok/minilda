if GCC
# These are good warnings to turn on by default
AM_CXXFLAGS = \
	-Wall \
	-Wcast-align \
	-Wcast-qual \
	-Wconversion \
	-Wextra \
	-Wfloat-equal \
	-Woverloaded-virtual \
	-Wparentheses \
	-Wpointer-arith \
	-Wswitch-enum \
	-Wwrite-strings
endif

AUTOMAKE_OPTIONS = no-dependencies

lib_LTLIBRARIES = libminilda.la
libminilda_la_SOURCES = \
	libminilda.cc \
	gibbs_sampler.cc

include_HEADERS = minilda.h

bin_PROGRAMS = lda_learn lda_infer
lda_learn_SOURCES = lda_learn.cc
lda_learn_LDADD = libminilda.la

lda_infer_SOURCES = lda_infer.cc
lda_infer_LDADD = libminilda.la

# unit tests using Google Test.
if HAVE_GTEST
check_PROGRAMS =

# unit tests you want to run when people type 'make check'.
TESTHARNESS = testharness.cc

TESTS =
endif
