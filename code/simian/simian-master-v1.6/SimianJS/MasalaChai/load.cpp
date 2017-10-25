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
#ifdef XP_WIN
# define PATH_MAX (MAX_PATH > _MAX_DIR ? MAX_PATH : _MAX_DIR)
# define getcwd _getcwd
const char PathSeparator = '\\';
#else
#include <libgen.h>
#include <unistd.h>
const char PathSeparator = '/';
#endif

static bool
IsAbsolutePath(const JSAutoByteString& filename) {
    const char* pathname = filename.ptr();

    if (pathname[0] == PathSeparator) return true;

#ifdef XP_WIN
    // On Windows there are various forms of absolute paths (see
    // http://msdn.microsoft.com/en-us/library/windows/desktop/aa365247%28v=vs.85%29.aspx
    // for details):
    //
    //   "\..."
    //   "\\..."
    //   "C:\..."
    //
    // The first two cases are handled by the test above so we only need a test
    // for the last one here.

    if ((strlen(pathname) > 3 && isalpha(pathname[0]) && pathname[1] == ':' && pathname[2] == '\\')) return true;
#endif

    return false;
}

/*
 * Resolve a (possibly) relative filename to an absolute path. If
 * |scriptRelative| is true, then the result will be relative to the directory
 * containing the currently-running script, or the current working directory if
 * the currently-running script is "-e" (namely, you're using it from the
 * command line.) Otherwise, it will be relative to the current working
 * directory.
 */
JSString*
ResolvePath(JSContext* cx, HandleString filenameStr, bool scriptRelativeMode) {
    JSAutoByteString filename(cx, filenameStr);
    if (!filename) return nullptr;

    if (IsAbsolutePath(filename)) return filenameStr;

    /* Get the currently executing script's name. */
    JS::AutoFilename scriptFilename;
    if (!DescribeScriptedCaller(cx, &scriptFilename)) return nullptr;

    if (!scriptFilename.get()) return nullptr;

    static char buffer[PATH_MAX+1];
    if (scriptRelativeMode) {
#ifdef XP_WIN
        // The docs say it can return EINVAL, but the compiler says it's void
        _splitpath(scriptFilename.get(), nullptr, buffer, nullptr, nullptr);
#else
        strncpy(buffer, scriptFilename.get(), PATH_MAX+1);
        if (buffer[PATH_MAX] != '\0') return nullptr;

        // dirname(buffer) might return buffer, or it might return a
        // statically-allocated string
        memmove(buffer, dirname(buffer), strlen(buffer) + 1);
#endif
    } else {
        const char* cwd = getcwd(buffer, PATH_MAX);
        if (!cwd) return nullptr;
    }

    size_t len = strlen(buffer);
    buffer[len] = '/';
    strncpy(buffer + len + 1, filename.ptr(), sizeof(buffer) - (len+1));
    if (buffer[PATH_MAX] != '\0') return nullptr;

    return JS_NewStringCopyZ(cx, buffer);
}

static bool
LoadScript(JSContext* cx, unsigned argc, Value* vp) {
    CallArgs args = CallArgsFromVp(argc, vp);
    bool scriptRelative = false;
    int endIndex = args.length();
    if (args[args.length()-1].isBoolean()) {
        scriptRelative = args[args.length()-1].toBoolean();
        endIndex--;
    }

    RootedString str(cx);
    for (unsigned i = 0; i < endIndex; i++) {
        str = JS::ToString(cx, args[i]);
        if (!str) {
            JS_ReportError(cx, "%s: Invalid argument", __FUNCTION__);
            return false;
        }
        str = ResolvePath(cx, str, scriptRelative);
        if (!str) {
            JS_ReportError(cx, "%s: Unable to resolve path", __FUNCTION__);
            return false;
        }
        JSAutoByteString filename(cx, str);
        if (!filename) return false;
        errno = 0;
        CompileOptions opts(cx);
        opts.setIntroductionType("chai load")
            .setUTF8(true)
            .setIsRunOnce(true)
            .setNoScriptRval(true);
        RootedScript script(cx);
        RootedValue unused(cx);
        if (!Evaluate(cx, opts, filename.ptr(), &unused)) {
            JS_ReportError(cx, "%s: Unable to evaluate the script %s", __FUNCTION__, filename.ptr());
            return false;
        }
    }

    args.rval().setUndefined();
    return true;
}
