
all: clean harris_test

harris_test:
	g++ -std=c++11 -o harris_test ./harris_test.cpp `pkg-config --cflags --libs opencv` -lboost_filesystem -lboost_system -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64

clean:
	rm -f ./harris_test