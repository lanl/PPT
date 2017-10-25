#Example:
#	SMJSDIR=$HOME/Work/JS/x86_64 MPIDIR=$HOME/Work/Simian/x86_64 make (OR)
#	SMJSDIR=$HOME/Work/JS/x86_64 MPIDIR=$HOME/Work/Lua/PDES/Lua/x86-64-mpich make (OR)
#	SMJSDIR=$HOME/Work/JS/x86_64 MPIDIR=$HOME/Work/JS/x86_64 make
CXX := g++
MAIN_50a1 := chai.50a1

all: ${MAIN_50a1}
	rm -f ./lib*.so ./lib*.dylib; cp ${SMJSDIR}/lib/libmozjs-50a1.dylib ${SMJSDIR}/lib/libnspr4.dylib ${SMJSDIR}/lib/libplc4.dylib ${SMJSDIR}/lib/libplds4.dylib .
	@#rm -f ./lib*.so ./lib*.dylib; find ${SMJSDIR}/lib \( -iname lib*.so -o -iname lib*.dylib \) -exec cp '{}' . \;
	@#rm -f ./lib*.so ./lib*.dylib; find ${SMJSDIR}/lib \( -iname lib*.so -o -iname lib*.dylib \) -exec ln -sf '{}' . \;

${MAIN_50a1}: ${MAIN_50a1}.cpp masala.cpp io.cpp load.cpp random.cpp mpi.cpp
	${CXX} -O3 -ffast-math -std=c++11 -Wno-invalid-offsetof -fwhole-program -o ${MAIN_50a1} ${MAIN_50a1}.cpp -I${SMJSDIR}/include/mozjs-50a1 -L${SMJSDIR}/lib -lmozjs-50a1 -I${MPIDIR}/include -L${MPIDIR}/lib -lmpi

${MAIN_50a1}_static: ${MAIN_50a1}.cpp masala.cpp io.cpp load.cpp random.cpp mpi.cpp
	${CXX} -O3 -ffast-math -std=c++11 -Wno-invalid-offsetof -fwhole-program -flto -funswitch-loops -fomit-frame-pointer -fassociative-math -ffinite-math-only -ffast-math -funroll-loops -fschedule-insns -fsched-spec-load -fsched-pressure -fselective-scheduling2 -fsel-sched-pipelining -fsel-sched-pipelining-outer-loops -o ${MAIN_50a1}_static ${MAIN_50a1}.cpp -I${SMJSDIR}/include/mozjs-50a1 -L${SMJSDIR}/lib -lmozjs-50a1 -I${MPIDIR}/include ${MPIDIR}/lib/libmpi.a  ${MPIDIR}/lib/libmpicxx.a ${MPIDIR}/lib/libmpifort.a ${MPIDIR}/lib/libpmpi.a -lpthread

clean:
	rm -rf ${MAIN_50a1} ${MAIN_50a1}_static ${MAIN_45} ${MAIN_45}_static lib*.*
