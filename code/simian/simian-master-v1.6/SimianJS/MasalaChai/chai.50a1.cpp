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
#include "simpleANSI.h"
#include <iostream>
#include <math.h>

#include "jsapi.h"
#include "jsprf.h"
#include <jsfriendapi.h> //For UInt8Array typed-array
#include "js/Initialization.h"
#include "js/Conversions.h"
#include <js/ProfilingFrameIterator.h>

using namespace std;
using namespace JS;
using namespace js;

#include "masala.cpp"

static const char tips[] = \
YEL "\n--------------\nTIPS FOR DEBUGGING YOUR CODE:\n\
To get a complete stack trace upon error you can try one or more of the following tips:\n\
1. Make sure that the errors thrown in your application are created using 'new Error(...)' syntax, and not just a string or other value.\n\
2. Run your application wrapped inside 'try ... catch ...' blocks as:\n\
    try { app(); } catch(err) { masala.io.print('Backtrace:\\n' + err.stack); throw(err); }\n\
3. Debug by adding either:\n\
    (a) masala.io.print(masala.dbg.getbt({args: true, locals: true, thisprops: true})); (or)\n\
    (b) masala.dbg.bt();\n\
    at places in your application code where you need a detailed backtrace.\n\
--------------\n\n" RST;

static const JSClassOps global_classOps = {
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr,
    JS_GlobalObjectTraceHook
};

// The basic class of the global object.
static const JSClass globalClass = {
    "global",
    JSCLASS_GLOBAL_FLAGS,
    &global_classOps
};

bool execBuffer(JSContext *cx, const char* bytes) {
    RootedValue v(cx);
    CompileOptions opts(cx);
    opts.setFileAndLine("<command line -e>", 1).setIsRunOnce(true); // Indicate source location for diagnostics
    return Evaluate(cx, opts, bytes, strlen(bytes), &v);
}

bool execFile(JSContext *cx, const char* filename) {
    RootedValue v(cx);
    CompileOptions opts(cx);
    return Evaluate(cx, opts, filename, &v);
}

// The error reporter callback.
void reportError(JSContext *cx, const char *message, JSErrorReport *report) {
    bool isWarning = JSREPORT_IS_WARNING(report->flags);
    bool isException = JSREPORT_IS_EXCEPTION(report->flags);
    bool isStrict = JSREPORT_IS_STRICT(report->flags);
    bool isStrictModeError = JSREPORT_IS_STRICT_MODE_ERROR(report->flags);

    fprintf(stderr, RED "Javascript %s%s @ " RST, \
            isWarning ? "Warning" : "Error", \
            isException ? " (Exception)" : "", isStrict ? " (Strict)" : "", \
            isStrictModeError ? " (Strict Mode Error)" : "");
    if (report->filename) fprintf(stderr, YEL ULINE "file:" RST " %s, " YEL ULINE "line:" RST " %d, " YEL ULINE "col:" RST " %d :>", report->filename, report->lineno, report->column);
    else fprintf(stderr, "<error location unavailable>");
    fprintf(stderr, RST "\n" BLINK CYN "  Message:" RST YEL BOLD " '%s'\n" RST, message);
}

static void
SetStandardCompartmentOptions(CompartmentOptions& options) {
    options.behaviors().setVersion(JSVERSION_DEFAULT);
    options.creationOptions().setSharedMemoryAndAtomicsEnabled(true);
}

int run(JSContext *cx, const char *source, bool isFile, int argc, const char *argv[]) {
    // Enter a request before running anything in the context.
    JSAutoRequest ar(cx);

    // Create the global object and a new compartment.
    RootedObject global(cx);
    CompartmentOptions compartmentOptions;
    SetStandardCompartmentOptions(compartmentOptions);
    global = JS_NewGlobalObject(cx, &globalClass, nullptr, FireOnNewGlobalHook, compartmentOptions);
    if (!global) return 1;

    // Enter the new global object's compartment.
    JS_EnterCompartment(cx, global);

    // Populate the global object with the standard globals, like Object and Array.
    if (!JS_InitStandardClasses(cx, global)) return 1;

    // Add customizations to the SpiderMokey engine.
    // Adds masala.dbg, masala.time, masala.random, masala.io, masala.mpi modules to the global namespace
    chai::addMasala(cx, global, argc, argv);

    // Some example source in a C string or a File. Larger, non-null-terminated buffers
    // can be used, if you pass the buffer length to Evaluate
    bool ok = isFile ? execFile(cx, source) : execBuffer(cx, source);
    if ((!ok) && (JS_IsExceptionPending(cx))) { // Catch exceptions thrown while executing the script/file
        RootedValue excn(cx); // Get exception object before printing and clearing exception.
        JS_GetPendingException(cx, &excn);
        ErrorReport report(cx);
        if (!report.init(cx, excn, ErrorReport::WithSideEffects)) {
            fprintf(stderr, "ERROR: Out of memory initializing ErrorReport\n");
            JS_ClearPendingException(cx);
            return 1;
        }

        reportError(cx, report.message(), report.report());

        // Display a concise stack-trace
        if (excn.isObject()) { // Try to access excn.trace, where excn is the thrown error which should be with Error() prototype
            JS::RootedObject excnObj(cx, &excn.toObject());
            bool found;
            if (!JS_HasProperty(cx, excnObj, "stack", &found)) {
                fprintf(stderr, "Thrown error is not an Error() object! No Backtrace found.\n%s", tips);
                JS_ClearPendingException(cx);
                return 1;
            } else { // found == true
                JS::RootedValue x(cx);
                if (!JS_GetProperty(cx, excnObj, "stack", &x)) {
                    fprintf(stderr, "Cannot retrieve Backtrace from the thrown error object!\n");
                    JS_ClearPendingException(cx);
                    return 1;
                }
                char *backTrace = JS_EncodeString(cx, x.toString());
                if (strlen(backTrace)) fprintf(stderr, WHT "\n|== " BOLD "STACK" BOLD " ==|\n" MAG);
                if (strlen(backTrace)) fprintf(stderr, "%s", backTrace);
                else fprintf(stderr, "  - empty -");
                fprintf(stderr, WHT "|===========|\n\n" RST);
                JS_free(cx, backTrace);
            }
        } else fprintf(stderr, "Thrown error is not an object! No Backtrace available.\n%s", tips);

        JS_ClearPendingException(cx);
    }

    // Leave the global object's compartment.
    JS_LeaveCompartment(cx, nullptr);

    return 0;
}

static const size_t gMaxStackSize = 2 * 128 * sizeof(size_t) * 1024; //TODO: Expose

int main(int argc, const char *argv[]) {
    if ((argc < 2) || (!strcmp(argv[1], "-h"))) {
        cerr << CYN BOLD BLINK "Usage: " RST WHT BOLD << argv[0] << " [-h | [-e 'JS-script'] | [-joff] file.js [args]]" RST << endl;
        exit(-1);
    }
    int argAdjust = 1, jitEnable = 1;
    if (!strcmp(argv[1], "-joff")) {
        argAdjust = 2;
        jitEnable = 0;
        chai::baselineJITEnabled = false;
        chai::ionJITEnabled = false;
    }

    // Initialize the JS engine.
    if (!JS_Init()) return 1;

    // Create a JS runtime.
    JSRuntime* rt = JS_NewRuntime(DefaultHeapMaxBytes, DefaultNurseryBytes); //Defualts are 32MB and 16MB. TODO: Expose
    if (!rt) return 1;
    SetWarningReporter(rt, reportError);

    JS_SetGCParameter(rt, JSGC_MAX_BYTES, 0xffffffff); //TODO: Expose
    JS_SetGCParameter(rt, JSGC_MODE, JSGC_MODE_INCREMENTAL); //TODO: Expose; TODO: Also allow manual garbagecollect()

    JS_SetNativeStackQuota(rt, gMaxStackSize); //TODO: Expose

    JS_SetGlobalJitCompilerOption(rt, JSJITCOMPILER_BASELINE_ENABLE, jitEnable);
    JS_SetGlobalJitCompilerOption(rt, JSJITCOMPILER_ION_ENABLE, jitEnable);
    chai::baselineJITEnabled = chai::ionJITEnabled = jitEnable;

    // Create a context.
    JSContext *cx = JS_GetContext(rt);
    if (!InitSelfHostedCode(cx)) return 1;

    // Run the JIT engine on the supplied script.
    int status = ((argc == 3) && (!strcmp(argv[1], "-e"))) ? \
                 run(cx, argv[2], false, 0, nullptr) : run(cx, argv[argAdjust], true, argc-argAdjust, argv+argAdjust);

    // Shut everything down.
    JS_DestroyRuntime(rt);
    JS_ShutDown();

    return status;
}
