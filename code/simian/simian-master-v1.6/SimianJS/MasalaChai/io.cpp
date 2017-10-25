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
static bool
Write(JSContext* cx, unsigned argc, Value* vp) {
    CallArgs args = CallArgsFromVp(argc, vp);

    for (unsigned i = 0; i < args.length(); i++) {
        JSString* str = ToString(cx, args[i]);
        if (!str) {
            JS_ReportError(cx, "%s: arg[%d] not a String", __FUNCTION__, i);
            return false;
        }
        char* bytes = JS_EncodeString(cx, str);
        if (!bytes) {
            JS_ReportError(cx, "%s: String at arg[%d] not encodable as bytes", __FUNCTION__, i);
            return false;
        }
        printf("%s%s", i ? " " : "", bytes);
        JS_free(cx, bytes);
    }

    fflush(stdout);
    args.rval().setNull();
    return true;
}

static bool
PrintLine(JSContext* cx, unsigned argc, Value* vp) {
    bool retVal = Write(cx, argc, vp);
    putchar('\n');
    fflush(stdout);
    return retVal;
}

// Use the fastest available getc.
#if defined(HAVE_GETC_UNLOCKED)
# define fast_getc getc_unlocked
#elif defined(HAVE__GETC_NOLOCK)
# define fast_getc _getc_nolock
#else
# define fast_getc getc
#endif

int
js_fgets(char* buf, int size, FILE* file) {
    int n, i, c;
    bool crflag;

    n = size - 1;
    if (n < 0) return -1;

    crflag = false;
    for (i = 0; i < n && (c = fast_getc(file)) != EOF; i++) {
        buf[i] = c;
        if (c == '\n') {        // any \n ends a line
            i++;                // keep the \n; we know there is room for \0
            break;
        }
        if (crflag) {           // \r not followed by \n ends line at the \r
            ungetc(c, file);
            break;              // and overwrite c in buf with \0
        }
        crflag = (c == '\r');
    }

    buf[i] = '\0';
    return i;
}

/*
 * function readline()
 * Provides a hook for scripts to read a line from stdin.
 */
static bool
ReadLine(JSContext* cx, unsigned argc, Value* vp) {
    CallArgs args = CallArgsFromVp(argc, vp);

#define BUFSIZE 256
    FILE* from = stdin;
    size_t buflength = 0;
    size_t bufsize = BUFSIZE;
    char* buf = (char*) JS_malloc(cx, bufsize);
    if (!buf) return false;

    bool sawNewline = false;
    size_t gotlength;
    while ((gotlength = js_fgets(buf + buflength, bufsize - buflength, from)) > 0) {
        buflength += gotlength;

        /* Are we done? */
        if (buf[buflength - 1] == '\n') {
            buf[buflength - 1] = '\0';
            sawNewline = true;
            break;
        } else if (buflength < bufsize - 1) {
            break;
        }

        /* Else, grow our buffer for another pass. */
        char* tmp;
        bufsize *= 2;
        if (bufsize > buflength) {
            tmp = static_cast<char*>(JS_realloc(cx, buf, bufsize / 2, bufsize));
        } else {
            JS_ReportOutOfMemory(cx);
            tmp = nullptr;
        }

        if (!tmp) {
            JS_free(cx, buf);
            return false;
        }

        buf = tmp;
    }

    /* Treat the empty string specially. */
    if (buflength == 0) {
        args.rval().set(feof(from) ? NullValue() : JS_GetEmptyStringValue(cx));
        JS_free(cx, buf);
        return true;
    }

    /* Shrink the buffer to the real size. */
    char* tmp = static_cast<char*>(JS_realloc(cx, buf, bufsize, buflength));
    if (!tmp) {
        JS_free(cx, buf);
        return false;
    }

    buf = tmp;

    /*
     * Turn buf into a JSString. Note that buflength includes the trailing null
     * character.
     */
    JSString* str = JS_NewStringCopyN(cx, buf, sawNewline ? buflength - 1 : buflength);
    JS_free(cx, buf);
    if (!str) return false;

    args.rval().setString(str);
    return true;
}

//Implements stdout/stderr/stdin in Open as integers 1, 2, -1
static bool
FOpen(JSContext* cx, unsigned argc, Value* vp) { // Experimental
    CallArgs args = CallArgsFromVp(argc, vp);
    // Guard against wrong arguments
    if ((args.length() == 1) && args[0].isNumber()) {
        int stream = args[0].toInt32();
        FILE *fp;
        switch(stream) {
            case -1:
                fp = stdin;
                break;
            case 1:
                fp = stdout;
                break;
            case 2:
                fp = stderr;
                break;
            default:
                args.rval().setUndefined(); //Error case
                return false;
        }
        JSObject *obj = JS_NewObject(cx, &masalaClass);
        JS_SetPrivate(obj, (void *)fp);
        args.rval().setObject(*obj);
        return true;
    }
    else if ((args.length() == 2) && args[0].isString() && args[1].isString()) {
        char *filename = JS_EncodeString(cx, args[0].toString());
        char *mode = JS_EncodeString(cx, args[1].toString());
        FILE* fp = fopen(filename, mode);
        JSObject *obj = JS_NewObject(cx, &masalaClass);
        JS_SetPrivate(obj, (void *)fp);
        args.rval().setObject(*obj);
        JS_free(cx, filename);
        JS_free(cx, mode);
        return true;
    }
    else {
        args.rval().setUndefined(); //Error case
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] is not a String or arg[1] not a String", __FUNCTION__);
        return false;
    }
}

static bool
FClose(JSContext* cx, unsigned argc, Value* vp) { // Experimental
    CallArgs args = CallArgsFromVp(argc, vp);
    // Guard against no arguments or a non-object arg0.
    if ((args.length() == 0) || (!args[0].isObject())) {
        args.rval().setInt32(-1);
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] not a file pointer", __FUNCTION__);
        return false;
    }
    else {
        FILE* fp = (FILE *)(JS_GetPrivate(&args[0].toObject()));
        int status = fclose(fp);
        if (status) {
            args.rval().setInt32(status); //Return Error# if any
            JS_ReportError(cx, "%s: Returned error status: %d", __FUNCTION__, status);
            return false;
        }
        else {
            args.rval().setInt32(status);
            return true;
        }
    }
}

static bool
FWrite(JSContext* cx, unsigned argc, Value* vp) {
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() == 0) || (!args[0].isObject())) {
        args.rval().setUndefined();
        return false;
    }
    else {
        FILE* fp = (FILE *)(JS_GetPrivate(&args[0].toObject()));
        for (unsigned i = 1; i < args.length(); i++) {
            JSString* str = ToString(cx, args[i]);
            if (!str) {
                args.rval().setUndefined();
                JS_ReportError(cx, "%s: arg[%d] not a string", __FUNCTION__, i);
                return false;
            }
            char* bytes = JS_EncodeString(cx, str);
            if (!bytes) {
                args.rval().setUndefined();
                JS_ReportError(cx, "%s: String at arg[%d] not encodable as bytes", __FUNCTION__, i);
                return false;
            }
            int status = fputs(bytes, fp);
            if (status == EOF) {
                args.rval().setUndefined();
                JS_ReportError(cx, "%s: fputs could not write string at arg[%d] to file", __FUNCTION__, i);
                return false;
            }
            JS_free(cx, bytes);
        }
        args.rval().setNull();
        fflush(fp);
        return true;
    }
}

static char*
fileReadLine(FILE *fp, long int &length) {
    char ch;
    int CUR_MAX = 4096;
    char *buffer = (char *)(malloc(sizeof(char) * CUR_MAX)); // allocate buffer.
    int count = 0; 
    length = 0;

    while ( (ch != '\n') && (ch != EOF) ) {
        if(length == CUR_MAX-1) { //Time to expand ?
            CUR_MAX <<= 1; //Expand to double the current size of anything similar.
            buffer = (char *)(realloc(buffer, sizeof(char) * CUR_MAX)); //Reallocate memory.
        }
        ch = getc(fp); //Read from stream.
        buffer[length] = ch; //Stuff in buffer.
        length++;
    }
    buffer[length] = '\0';
    return buffer;
}

static bool
FRead(JSContext* cx, unsigned argc, Value* vp) { //fp, "a"/"l"/number
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 2) || (!args[0].isObject()) || !(args[1].isString() || args[1].isNumber())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] is not an Object or arg[1] not a String/Number", __FUNCTION__);
        return false;
    }
    else {
        FILE* fp = (FILE *)(JS_GetPrivate(&args[0].toObject()));
        if (args[1].isString()) {
            char* typeStr = JS_EncodeString(cx, args[1].toString());
            if (!strcmp(typeStr, "a")) { //Read whole file
                fseek(fp, 0, SEEK_END);
                long fsize = ftell(fp);
                fseek(fp, 0, SEEK_SET);  //same as rewind(f);

                char *string = (char *)(malloc(sizeof(char) * (fsize + 1)));
                size_t length = fread(string, 1, fsize, fp);
                string[length] = '\0';
                args.rval().setString(JS_NewStringCopyN(cx, string, length+1));
                free(string);
            }
            else if (!strcmp(typeStr, "l")) { //Read one line
                long int length;
                char *string = fileReadLine(fp, length);
                args.rval().setString(JS_NewStringCopyN(cx, string, length));
                free(string);
            }
            args.rval().setNull();
            JS_free(cx, typeStr);
        }
        else if (args[1].isNumber()) {
            int numBytes = args[1].toNumber();
            if (numBytes > 0) {
                char *string = (char *)(malloc(sizeof(char) * numBytes));
                size_t length = fread(string, 1, numBytes, fp);
                args.rval().setString(JS_NewStringCopyN(cx, string, length));
                free(string);
            }
            else args.rval().setNull();
        }
        return true;
    }
}


static bool
FSeek(JSContext* cx, unsigned argc, Value* vp) { //fp, whence("set"/"cur"/"end"), offset
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 3) || (!args[0].isObject()) || (!args[1].isString()) || (!args[2].isNumber())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] is not an Object or arg[1] not a String or arg[2] not a Number", __FUNCTION__);
        return false;
    }
    else {
        FILE* fp = (FILE *)(JS_GetPrivate(&args[0].toObject()));
        char* whenceStr = JS_EncodeString(cx, args[1].toString());
        if (strcmp(whenceStr, "set") && strcmp(whenceStr, "cur") && strcmp(whenceStr, "end")) {
            args.rval().setUndefined();
            JS_ReportError(cx, "%s: arg[1] should be one of \"set\", \"cur\", \"end\"", __FUNCTION__);
            return false;
        }
        int whence = (!strcmp(whenceStr, "set") ? SEEK_SET : (!strcmp(whenceStr, "end") ? SEEK_END : SEEK_CUR));
        int64_t offset = (int64_t)(args[2].toNumber());
        fseek(fp, offset, whence);
        args.rval().setNull();
        JS_free(cx, whenceStr);
        return true;
    }
}

static bool
FTell(JSContext* cx, unsigned argc, Value* vp) { //fp
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() == 0) || (!args[0].isObject())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] is not an Object", __FUNCTION__);
        return false;
    }
    else {
        FILE* fp = (FILE *)(JS_GetPrivate(&args[0].toObject()));
        double offset = (double)(ftell(fp));
        args.rval().setNumber(offset);
        return true;
    }
}

static bool
FFlush(JSContext* cx, unsigned argc, Value* vp) { //fp
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() == 0) || (!args[0].isObject())) {
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] is not an Object", __FUNCTION__);
        args.rval().setInt32(-1);
        return false;
    }
    else {
        FILE* fp = (FILE *)(JS_GetPrivate(&args[0].toObject()));
        int status = fflush(fp);
        args.rval().setInt32(status);
        return (status ? false : true);
    }
}

static bool
OSExit(JSContext* cx, unsigned argc, Value* vp) { //exit code
    CallArgs args = CallArgsFromVp(argc, vp);
    exit(((args.length() == 0) || (!args[0].isNumber())) ? 0 : args[0].toInt32());
    return false;
}
