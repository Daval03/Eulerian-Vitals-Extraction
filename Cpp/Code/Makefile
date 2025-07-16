# Busca todos los archivos .cpp y genera nombres de ejecutables
SRCS = $(wildcard *.cpp)
EXECUTABLES = $(SRCS:.cpp=)


main:
	g++ Vital_Estimator.cpp Config.h Signal_Processing.cpp Utils_Process_Data.cpp Face_Detector.cpp \
    -o Vital_Estimator `pkg-config --cflags --libs opencv4` -I/usr/include/eigen3 -lfftw3 -O2

	./Vital_Estimator

delete:
	rm -f $(EXECUTABLES)

#Memory
mem:
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./Vital_Estimator
