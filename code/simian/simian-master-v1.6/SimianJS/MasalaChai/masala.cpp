/*
Copyright (c) 2015, Los Alamos National Security, LLC
All rights reserved.

Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/*
Author: Nandakishore Santhi
Date: 23 May, 2016
Modified: 07 Dec, 2016
Copyright: Open source, must acknowledge original author
*/
/* ADD USER DEFINED FUNCTIONS */
namespace chai {
    static const JSClass masalaClass = {
        "masala",
        // Flag JSCLASS_HAS_PRIVATE: Can hold a private data. See:
        // void JS_SetPrivate(JSObject *obj, void *data);
        // void * JS_GetPrivate(JSObject *obj);
        // Value.setObject(JSObject &); Value.toObject();
        // Value.isObject(); Value ObjectValue(JSObject& obj);
        JSCLASS_HAS_PRIVATE
    };

#include "load.cpp"
#include "io.cpp"
#include "jit.cpp"
#include "time.cpp"
#include "dbg.cpp"
#include "random.cpp"
#include "mpi.cpp"

    bool addMasala(JSContext* cx, HandleObject global, const int argc, const char *argv[]) {
        RootedObject objMasala(cx);
        objMasala = JS_NewObject(cx, &masalaClass);

        RootedObject objRandom(cx);
        objRandom = JS_NewObject(cx, &masalaClass);
        if (objRandom) {
            if (!JS_DefineFunction(cx, objRandom, "seed", (JSNative) SRand, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objRandom, "rand", (JSNative) Rand, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objRandom, "uniform", (JSNative) Random, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objRandom, "range", (JSNative) RandomRange, 0, 0)) return false;

            if (!JS_DefineProperty(cx, objMasala, "random", objRandom, 0)) return false;
        }
        else return false;

        RootedObject objIO(cx);
        objIO = JS_NewObject(cx, &masalaClass);
        if (objIO) {
            if (!JS_DefineFunction(cx, objIO, "load", (JSNative) LoadScript, 0, 0)) return false;

            if (!JS_DefineFunction(cx, objIO, "print", (JSNative) PrintLine, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objIO, "write", (JSNative) Write, 0, 0)) return false;

            if (!JS_DefineFunction(cx, objIO, "readline", (JSNative) ReadLine, 0, 0)) return false;

            if (!JS_DefineFunction(cx, objIO, "fopen", (JSNative) FOpen, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objIO, "fclose", (JSNative) FClose, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objIO, "fwrite", (JSNative) FWrite, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objIO, "fread", (JSNative) FRead, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objIO, "fseek", (JSNative) FSeek, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objIO, "fflush", (JSNative) FFlush, 0, 0)) return false;

            if (!JS_DefineFunction(cx, objIO, "exit", (JSNative) OSExit, 0, 0)) return false;

            //Add the JS property masala.io.arg, which is a JS array with the command-line arguments in it
            AutoValueVector args(cx);
            for (int i=0; i<argc; i++) {
                RootedValue v(cx);
                RootedString arg_str(cx, JS_NewStringCopyZ(cx, argv[i]));
                v.setString(arg_str);
                if (!args.append(v)) return false;
            }
            RootedObject argArray(cx);
            argArray = JS_NewArrayObject(cx, args);
            if (!JS_DefineProperty(cx, objIO, "arg", argArray, 0)) return false;

            if (!JS_DefineProperty(cx, objMasala, "io", objIO, 0)) return false;
        }
        else return false;

        RootedObject objJIT(cx);
        objJIT = JS_NewObject(cx, &masalaClass);
        if (objJIT) {
            if (!JS_DefineFunction(cx, objJIT, "on", (JSNative) JITOn, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objJIT, "off", (JSNative) JITOff, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objJIT, "status", (JSNative) JITStatus, 0, 0)) return false;

            if (!JS_DefineProperty(cx, objMasala, "jit", objJIT, 0)) return false;
        }
        else return false;

        RootedObject objTime(cx);
        objTime = JS_NewObject(cx, &masalaClass);
        if (objTime) {
            if (!JS_DefineFunction(cx, objTime, "now", (JSNative) Now, 0, 0)) return false;

            if (!JS_DefineProperty(cx, objMasala, "time", objTime, 0)) return false;
        }
        else return false;

        RootedObject objDbg(cx);
        objDbg = JS_NewObject(cx, &masalaClass);
        if (objTime) {
            if (!JS_DefineFunction(cx, objDbg, "bt", (JSNative) DumpBacktrace, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objDbg, "getbt", (JSNative) GetBacktrace, 0, 0)) return false;

            if (!JS_DefineProperty(cx, objMasala, "dbg", objDbg, 0)) return false;
        }
        else return false;

        RootedObject objMPI(cx);
        objMPI = JS_NewObject(cx, &masalaClass);
        if (objMPI) {
            if (!JS_DefineFunction(cx, objMPI, "init", (JSNative) MPIInit, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "finalize", (JSNative) MPIFinalize, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "barrier", (JSNative) MPIBarrier, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "rank", (JSNative) MPIRank, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "size", (JSNative) MPISize, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "iprobe", (JSNative) MPIIProbe, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "iprobetrials", (JSNative) MPIIProbeTrials, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "probe", (JSNative) MPIProbe, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "send", (JSNative) MPISend, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "isend", (JSNative) MPIISend, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "getcount", (JSNative) MPIGetCount, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "getelements", (JSNative) MPIGetElements, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "recv", (JSNative) MPIRecv, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "allreduce", (JSNative) MPIAllReduce, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "alltoallSum", (JSNative) MPIAlltoallSum, 0, 0)) return false;
            if (!JS_DefineFunction(cx, objMPI, "sendAndCount", (JSNative) MPISendAndCount, 0, 0)) return false;

            if (!JS_DefineProperty(cx, objMasala, "mpi", objMPI, 0)) return false;
        }
        else return false;

        if (!JS_DefineProperty(cx, global, "masala", objMasala, 0)) return false;
        return true;
    }
}
