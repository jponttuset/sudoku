CFLAGS = -std=c++11
LIBS = -ltesseract -llept -lopencv_highgui -lopencv_imgproc -lopencv_core

sudoku_ar_solver: sudoku_ar_solver.o
	g++ -o $@ $< $(LIBS)

clean:
	rm -f *.o

.cpp.o:
	g++ -c $(CFLAGS) -o $@ $<
