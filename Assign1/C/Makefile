CC      = mpicc
CFLAGS  = -O2 -Wall
LIBS    = -llapack -lblas -lm

TARGETS = tsqr tsqr3

all: $(TARGETS)

# Q2
tsqr: tsqr.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Q3
tsqr3: tsqr3.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

run2: tsqr
	mpirun -np 4 ./tsqr

# Example: make run3 M=40000 N=50
run3: tsqr3
	mpirun -np 4 ./tsqr3 $(M) $(N)

clean:
	rm -f $(TARGETS)