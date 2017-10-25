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
static bool baselineJITEnabled = false;
static bool ionJITEnabled = false;

static bool
JITOn(JSContext* cx, unsigned argc, Value* vp) {
    JSRuntime *rt = JS_GetRuntime(cx);
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 1) || (!args[0].isString())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: arg[0] is not a String", __FUNCTION__);
        return false;
    }

    /*
    // Throw if disabling the JITs and there's JIT code on the stack, to avoid
    // assertion failures.
    jit::JitProfilingFrameIterator iter();
    if (!iter.done()) {
        JS_ReportError(cx, "Can't turn off JITs with JIT code on the stack.");
        return false;
    }
    */

    char* optStr = JS_EncodeString(cx, args[0].toString());
    if (!strcmp(optStr, "baseline")) JS_SetGlobalJitCompilerOption(rt, JSJITCOMPILER_BASELINE_ENABLE, true);
    else if (!strcmp(optStr, "ion")) JS_SetGlobalJitCompilerOption(rt, JSJITCOMPILER_ION_ENABLE, true);

    baselineJITEnabled = true;
    ionJITEnabled = true;

    args.rval().setUndefined();
    return true;
}

static bool
JITOff(JSContext* cx, unsigned argc, Value* vp) {
    JSRuntime *rt = JS_GetRuntime(cx);
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 1) || (!args[0].isString())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: arg[0] is not a String", __FUNCTION__);
        return false;
    }

    /*
    // Throw if disabling the JITs and there's JIT code on the stack, to avoid
    // assertion failures.
    jit::JitProfilingFrameIterator iter();
    if (!iter.done()) {
        JS_ReportError(cx, "Can't turn off JITs with JIT code on the stack.");
        return false;
    }
    */

    char* optStr = JS_EncodeString(cx, args[0].toString());
    if (!strcmp(optStr, "baseline")) JS_SetGlobalJitCompilerOption(rt, JSJITCOMPILER_BASELINE_ENABLE, false);
    else if (!strcmp(optStr, "ion")) JS_SetGlobalJitCompilerOption(rt, JSJITCOMPILER_ION_ENABLE, false);

    baselineJITEnabled = false;
    ionJITEnabled = false;

    args.rval().setUndefined();
    return true;
}

static bool
JITStatus(JSContext* cx, unsigned argc, Value* vp) {
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 1) || (!args[0].isString())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: arg[0] is not a String", __FUNCTION__);
        return false;
    }

    char* optStr = JS_EncodeString(cx, args[0].toString());
    if (!strcmp(optStr, "baseline")) args.rval().setBoolean(baselineJITEnabled);
    else if (!strcmp(optStr, "ion")) args.rval().setBoolean(ionJITEnabled);

    return true;
}
