# CMSC416 A4 Makefile
AN = a4
CLASS = 416
GS_COURSE_ID = 1239306
GS_ASSIGN_ID = 8027859

COPTS = -g -Wall -O3 -fstack-protector-all -Wno-unused-variable -Wno-unused-result
GCC = gcc $(COPTS)
OMPCC = $(GCC) -fopenmp
NVCCOPTS = --compiler-options -g,-Wall,-fstack-protector-all,-Wno-unused-variable,-Wno-unused-result
# suppress some diagnostic warnings like csating char* variables
NVCCSUPPRESS = --diag-suppress 1650,2464
NVCC = nvcc -g -O3 $(NVCCOPTS) $(NVCCSUPPRESS)
.SHELLFLAGS = -O nullglob -c
SHELL  = /bin/bash
CWD    = $(shell pwd | sed 's/.*\///g')

ifneq (,$(findstring zaratan,$(shell hostname)))
# zaratan BLAS/LAPACK library
	CBLAS_LIB = -lopenblas
else ifeq ($(GRADESCOPE), 1)
# gradescope BLAS/LAPACK library
	CBLAS_LIB = -lopenblas
else
# adjust this as needed to suit your home system
	CBLAS_LIB = -lopenblas_64
#	CBLAS_LIB = -lcblas
endif

# openblas creates a dependency on gomp/threaded implementation which
# requires suppressing some valgrind errors and ensuring that
# OMP_NUM_THREADS=1 when running to prevent automatic multithreading




export PARALLEL?=False		# disable parallel testing if not overridden

PROGRAMS = \
	kmeans_serial \
	kmeans_omp \
	kmeans_cuda \
	kmeans_blas

all : $(PROGRAMS)

# cleaning target to remove compiled programs/objects
clean :
	rm -f $(PROGRAMS) *.o vgcore.*

clean-tests :
	rm -rf test-results outdirs

help :
	@echo 'Typical usage is:'
	@echo '  > make                          # build all programs'
	@echo '  > make clean                    # remove all compiled items'
	@echo '  > make clean-tests              # remove all temporary testing files'
	@echo '  > make zip                      # create a zip file for submission'
	@echo '  > make prob1                    # built targets associated with problem 1'
	@echo '  > make test-setup               # prepare for testing, download data files'
	@echo '  > make test                     # run all tests'
	@echo '  > make test-prob2               # run test for problem 2'
	@echo '  > make test-prob2 testnum=5     # run problem 2 test #5 only'
	@echo '  > make update                   # download and install any updates to project files'
	@echo '  > make submit                   # upload submission to Gradescope'

############################################################
# 'make zip' to create complete.zip for submission
ZIPNAME = $(AN)-complete.zip
ZIP_OPTIONS = --exclude \*/sample-mnist-data/\*
zip : clean clean-tests
	rm -f $(ZIPNAME)
	cd .. && zip "$(CWD)/$(ZIPNAME)" -r "$(CWD)" $(ZIP_OPTIONS)
	@echo Zip created in $(ZIPNAME)
	@if (( $$(stat -c '%s' $(ZIPNAME)) > 10*(2**20) )); then echo "WARNING: $(ZIPNAME) seems REALLY big, check there are no abnormally large test files"; du -h $(ZIPNAME); fi
	@if (( $$(unzip -t $(ZIPNAME) | wc -l) > 256 )); then echo "WARNING: $(ZIPNAME) has 256 or more files in it which may cause submission problems"; fi
	@echo


############################################################
# `make update` to get project updates
update :
ifeq ($(findstring solution,$(CWD)),)
	curl -s https://www.cs.umd.edu/~profk/$(CLASS)/$(AN)-update.sh | /bin/bash 
else
	@echo "Cowardly refusal to update solution"
endif

################################################################################
# `make submit` to upload to gradescope
submit : zip 
	@chmod u+x gradescope-submit
	@echo '=== SUBMITTING TO GRADESCOPE ==='
	./gradescope-submit $(GS_COURSE_ID) $(GS_ASSIGN_ID) $(ZIPNAME)

################################################################################
# testing targets
test : test-prob1 test-prob2

# data files that are downloaded for testing; large so do not include
# in distribution or submission
DATA_NAME=sample-mnist-data
DATA_FILE="$(DATA_NAME).zip"
DATA_URL="https://www.cs.umd.edu/~profk/416/$(DATA_FILE)"

test-setup:
	@chmod u+x testy
	@if [[ ! -x $(DATA_NAME) ]]; then \
	  wget $(DATA_URL); \
	  unzip $(DATA_FILE); \
	  rm $(DATA_FILE);    \
	fi

################################################################################
# kmeans_serial
kmeans_serial : kmeans_serial.c kmeans_util.c
	$(GCC) -o $@ $^ -lm

test-serial: test-setup kmeans_serial
	./testy test_kmeans_serial.org $(testnum)

################################################################################
# Problem 1 : kmeans_omp
prob1: kmeans_omp

kmeans_omp : kmeans_omp.c kmeans_util.c
	$(OMPCC) -o $@ $^ -lm

test-prob1: test-setup prob1 test_kmeans_omp.org
	./testy test_kmeans_omp.org $(testnum)


################################################################################
# Problem 2 : kmeans_cuda
prob2: kmeans_cuda

# comment this line if you do not have any support files for kmeans_cuda
kmeans_util.cu :
	$(error "No kmeans_util.cu found; try symlinking to create it as in 'ln -s kmeans_util.c kmeans_util.cu'")

kmeans_cuda : kmeans_cuda.cu kmeans_util.cu
	$(NVCC) -o $@ $^ -lm

test-prob2: test-setup prob2 test_kmeans_cuda.org
	./testy test_kmeans_cuda.org $(testnum)

################################################################################
# Problem 3 : kmeans_blas (MAKEUP)
prob3: kmeans_blas

kmeans_blas : kmeans_blas.c kmeans_util.c
	$(GCC) -o $@ $^ -lm $(CBLAS_LIB)

test-prob3: test-setup prob3 test_kmeans_blas.org
	./testy test_kmeans_blas.org $(testnum)

