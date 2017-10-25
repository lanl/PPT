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
#include <mpi.h>

static int32_t *Simian_sndCounts, *Simian_rcvCounts;
static int Simian_numRanks = 0;

static bool
MPIInit(JSContext* cx, unsigned argc, Value* vp) { //IN: , OUT: boolean status
    CallArgs args = CallArgsFromVp(argc, vp);

    if (MPI_Init(nullptr, nullptr) == MPI_SUCCESS) {
    	if (MPI_Comm_size(MPI_COMM_WORLD, &Simian_numRanks) == MPI_SUCCESS) {
	    Simian_sndCounts = (int32_t*)calloc(Simian_numRanks, sizeof(int32_t));
	    Simian_rcvCounts = (int32_t*)calloc(Simian_numRanks, sizeof(int32_t));
	}
        args.rval().setBoolean(true);
        return true;
    }
    else {
        args.rval().setBoolean(false);
        JS_ReportError(cx, "%s: MPI_Init was unsuccessful", __FUNCTION__);
        return false;
    }
}

static bool
MPIFinalize(JSContext* cx, unsigned argc, Value* vp) { //IN: , OUT: boolean status
    CallArgs args = CallArgsFromVp(argc, vp);

    if (MPI_Finalize() == MPI_SUCCESS) {
	free(Simian_sndCounts);
	free(Simian_rcvCounts);
        args.rval().setBoolean(true);
        return true;
    }
    else {
        args.rval().setBoolean(false);
        JS_ReportError(cx, "%s: MPI_Finalize was unsuccessful", __FUNCTION__);
        return false;
    }
}

static bool
MPIBarrier(JSContext* cx, unsigned argc, Value* vp) { //IN: , OUT: boolean status
    CallArgs args = CallArgsFromVp(argc, vp);

    if (MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS) {
        args.rval().setBoolean(true);
        return true;
    }
    else {
        args.rval().setBoolean(false);
        JS_ReportError(cx, "%s: MPI_Barrier was unsuccessful", __FUNCTION__);
        return false;
    }
}

static bool
MPIRank(JSContext* cx, unsigned argc, Value* vp) { //IN: , OUT: int rank
    CallArgs args = CallArgsFromVp(argc, vp);

    int rank;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS) {
        args.rval().setInt32(rank);
        return true;
    }
    else {
        args.rval().setInt32(-1);
        JS_ReportError(cx, "%s: MPI_Comm_rank was unsuccessful", __FUNCTION__);
        return false;
    }
}

static bool
MPISize(JSContext* cx, unsigned argc, Value* vp) { //IN: , OUT: int size
    CallArgs args = CallArgsFromVp(argc, vp);

    int size;
    if (MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS) {
        args.rval().setInt32(size);
        return true;
    }
    else {
        args.rval().setInt32(-1);
        JS_ReportError(cx, "%s: MPI_Comm_size was unsuccessful", __FUNCTION__);
        return false;
    }
}

static bool
MPIIProbe(JSContext* cx, unsigned argc, Value* vp) { //Non-blocking //IN: int src[null==ANY], int tag[null==ANY], OUT: boolean/undefined status
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 2) || !(args[0].isNumber() || args[0].isNull()) || !(args[1].isNumber() || args[1].isNull())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] (src) is not Number/Null or arg[1] (tag) is not Number/Null", __FUNCTION__);
        return false;
    }
    else {
        int val;
        MPI_Status status;
        int32_t src = args[0].isNull() ? MPI_ANY_SOURCE : (int32_t)(args[0].toNumber());
        int32_t tag = args[1].isNull() ? MPI_ANY_TAG  : (int32_t)(args[1].toNumber());

        if (MPI_Iprobe(src, tag, MPI_COMM_WORLD, &val, &status) == MPI_SUCCESS) {
            args.rval().setBoolean(val > 0);
            return true;
        }
        else {
            args.rval().setUndefined();
            JS_ReportError(cx, "%s: MPI_Iprobe was unsuccessful", __FUNCTION__);
            return false;
        }
    }
}

static bool
MPIIProbeTrials(JSContext* cx, unsigned argc, Value* vp) { //Non-blocking //IN: int trials[null==1], int src[null==ANY], int tag[null==ANY], OUT: boolean/undefined status
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 3) || !(args[0].isNumber() || args[0].isNull()) || !(args[1].isNumber() || args[1].isNull()) || !(args[2].isNumber() || args[2].isNull())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] (numTrials) is not Number/Null or arg[1] (src) is not Number/Null or arg[2] (tag) is not Number/Null", __FUNCTION__);
        return false;
    }
    else {
        int val;
        MPI_Status status;
        int32_t trials = args[0].isNull() ? 1 : (int32_t)(args[0].toNumber());
        int32_t src = args[1].isNull() ? MPI_ANY_SOURCE : (int32_t)(args[1].toNumber());
        int32_t tag = args[2].isNull() ? MPI_ANY_TAG  : (int32_t)(args[2].toNumber());

        for (int i=0; i<trials; i++) { //Probe trial number of times (to catch more difficult cases)
            if (MPI_Iprobe(src, tag, MPI_COMM_WORLD, &val, &status) == MPI_SUCCESS) {
                if (val > 0) {
                    args.rval().setBoolean(true);
                    return true;
                }
            }
            else {
                args.rval().setUndefined();
                JS_ReportError(cx, "%s: MPI_Iprobe was unsuccessful", __FUNCTION__);
                return false;
            }
        }
        args.rval().setBoolean(false);
        return true;
    }
}

static bool
MPIProbe(JSContext* cx, unsigned argc, Value* vp) { //Blocking //IN: int src[null==ANY], int tag[null==ANY], OUT: boolean/undefined status
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 2) || !(args[0].isNumber() || args[0].isNull()) || !(args[1].isNumber() || args[1].isNull())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] (src) is not Int32/Null or arg[1] (tag) is not Int32/Null", __FUNCTION__);
        return false;
    }
    else {
        MPI_Status status;
        int32_t src = args[0].isNull() ? MPI_ANY_SOURCE : (int32_t)(args[0].toNumber());
        int32_t tag = args[1].isNull() ? MPI_ANY_TAG  : (int32_t)(args[1].toNumber());

        bool val = (MPI_Probe(src, tag, MPI_COMM_WORLD, &status) == MPI_SUCCESS);
        args.rval().setBoolean(val);
        return true;
    }
}

static bool
MPISend(JSContext* cx, unsigned argc, Value* vp) { //Blocking //IN: ArrayBuffer message, int dst, int tag[null==msgLength], OUT: boolean/undefined status
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 3) || !(args[0].isObject()) || !(args[1].isNumber()) || !(args[2].isNumber() || args[2].isNull())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] (message) is not an Object or arg[1] (dst) is not Int32 or arg[2] (tag) is not Int32/Null", __FUNCTION__);
        return false;
    }
    else {
        AutoCheckCannotGC nogc;
        RootedObject messageObj(cx, &args[0].toObject());
        if (JS_IsArrayBufferObject(messageObj)) {
            uint32_t length = JS_GetArrayBufferByteLength(messageObj);
            void *buffer = JS_StealArrayBufferContents(cx, messageObj);
            int32_t dst = (int32_t)(args[1].toNumber());
            int32_t tag = args[2].isNull() ? length : (int32_t)(args[2].toNumber());

            bool retVal;
            if (MPI_Send((void *)buffer, length, MPI_BYTE, dst, tag, MPI_COMM_WORLD) == MPI_SUCCESS) {
                args.rval().setBoolean(true);
                retVal = true;
            }
            else {
                args.rval().setBoolean(false);
                JS_ReportError(cx, "%s: MPI_Send was unsuccessful", __FUNCTION__);
                retVal = false;
            }
            free(buffer);
            return retVal;
        }
        else {
            args.rval().setUndefined();
            JS_ReportError(cx, "%s: arg[0] (message) was not an ArrayBuffer Object", __FUNCTION__);
            return false;
        }
    }
}

static bool
MPIISend(JSContext* cx, unsigned argc, Value* vp) { //Non-blocking //IN: ArrayBuffer message, int dst[null==ANY], int tag[null==ANY], OUT: boolean/undefined status
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 3) || !(args[0].isObject()) || !(args[1].isNumber()) || !(args[2].isNumber() || args[2].isNull())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] (message) is not an Object or arg[1] (dst) is not Int32 or arg[2] (tag) is not Int32/Null", __FUNCTION__);
        return false;
    }
    else {
        AutoCheckCannotGC nogc;
        MPI_Request request;
        RootedObject messageObj(cx, &args[0].toObject());
        if (JS_IsArrayBufferObject(messageObj)) {
            uint32_t length = JS_GetArrayBufferByteLength(messageObj);
            void *buffer = JS_StealArrayBufferContents(cx, messageObj);
            int32_t dst = (int32_t)(args[1].toNumber());
            int32_t tag = args[2].isNull() ? length : (int32_t)(args[2].toNumber());

            bool retVal;
            if (MPI_Isend((void *)buffer, length, MPI_BYTE, dst, tag, MPI_COMM_WORLD, &request) == MPI_SUCCESS) {
                args.rval().setBoolean(true);
                retVal = true;
            }
            else {
                args.rval().setBoolean(false);
                JS_ReportError(cx, "%s: MPI_Isend was unsuccessful", __FUNCTION__);
                retVal = false;
            }
            return retVal;
        }
        else {
            args.rval().setUndefined();
            JS_ReportError(cx, "%s: arg[0] (message) was not an ArrayBuffer Object", __FUNCTION__);
            return false;
        }
    }
}

static bool
MPIGetCount(JSContext* cx, unsigned argc, Value* vp) { //IN: , OUT: int/undefined count
    CallArgs args = CallArgsFromVp(argc, vp);
    int src = MPI_ANY_SOURCE;
    int tag = MPI_ANY_TAG;

    MPI_Status status;
    int val;
    if (MPI_Iprobe(src, tag, MPI_COMM_WORLD, &val, &status) != MPI_SUCCESS) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: MPI_Iprobe for status fetch was unsuccessful", __FUNCTION__);
        return false;
    }

    int count;
    if (MPI_Get_count(&status, MPI_BYTE, &count) == MPI_SUCCESS) {
        args.rval().setInt32(count);
        return true;
    }
    else {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: MPI_Get_count was unsuccessful", __FUNCTION__);
        return false;
    }
}

static bool
MPIGetElements(JSContext* cx, unsigned argc, Value* vp) { //Alternative to GetCount //IN: , OUT: int/undefined count
    CallArgs args = CallArgsFromVp(argc, vp);
    int src = MPI_ANY_SOURCE;
    int tag = MPI_ANY_TAG;

    MPI_Status status;
    int val;
    if (MPI_Iprobe(src, tag, MPI_COMM_WORLD, &val, &status) != MPI_SUCCESS) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: MPI_Iprobe for status fetch was unsuccessful", __FUNCTION__);
        return false;
    }

    int count;
    if (MPI_Get_elements(&status, MPI_BYTE, &count) == MPI_SUCCESS) {
        args.rval().setInt32(count);
        return true;
    }
    else {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: MPI_Get_elements was unsuccessful", __FUNCTION__);
        return false;
    }
}

static bool
MPIRecv(JSContext* cx, unsigned argc, Value* vp) { //Blocking //IN: int maxSize[null==autodetect], int src[null==ANY], int tag[null==ANY], OUT: ArrayBuffer/undefined message
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 3) || !(args[0].isNumber() || args[0].isNull()) || !(args[1].isNumber() || args[1].isNull()) || !(args[2].isNumber() || args[2].isNull())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] (maxSize) is not Int32/Null or arg[1] (src) is not Int32/Null or arg[2] (tag) is not Int32/Null", __FUNCTION__);
        return false;
    }
    else {
        int32_t src = args[1].isNull() ? MPI_ANY_SOURCE : (int32_t)(args[1].toNumber());
        int32_t tag = args[2].isNull() ? MPI_ANY_TAG  : (int32_t)(args[2].toNumber());

        MPI_Status status;
        int32_t maxSize;

        if (args[0].isNull()) { //maxSize is not implicitly given so find using Get_count()
            int val;
            if (MPI_Iprobe(src, tag, MPI_COMM_WORLD, &val, &status) != MPI_SUCCESS) {
                args.rval().setUndefined();
                JS_ReportError(cx, "%s: MPI_Iprobe was unsuccessful", __FUNCTION__);
                return false;
            }

            if ((MPI_Get_count(&status, MPI_BYTE, &maxSize) != MPI_SUCCESS) || (maxSize == MPI_UNDEFINED)) {
                args.rval().setUndefined();
                JS_ReportError(cx, "%s: MPI_Get_count was unsuccessful", __FUNCTION__);
                return false;
            }
        }
        else maxSize = (int32_t)(args[0].toNumber());

        uint8_t *buffer = (uint8_t *)malloc(maxSize * sizeof(uint8_t));
        if (MPI_Recv((void *)buffer, maxSize, MPI_BYTE, src, tag, MPI_COMM_WORLD, &status) == MPI_SUCCESS) {
            JSObject* messageObj = JS_NewArrayBufferWithContents(cx, maxSize, buffer);
            args.rval().setObject(*messageObj);
            return true;
        }
        else {
            args.rval().setUndefined();
            JS_ReportError(cx, "%s: MPI_Recv was unsuccessful", __FUNCTION__);
            return false;
        }
    }
}

static bool
MPIAllReduce(JSContext* cx, unsigned argc, Value* vp) { //IN: double partial, int reduction_op[0==MIN,else==SUM], OUT: double/undefined result
    const int32_t OP_MIN = 0;

    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 2) || !(args[0].isNumber()) || !(args[1].isNumber())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] (partial) is not Number or arg[1] (op) is not Int32", __FUNCTION__);
        return false;
    }
    else {
        double partial = args[0].toNumber();
        int32_t op = (int32_t)(args[1].toNumber());
        double result;
        
        if (MPI_Allreduce(&partial, &result, 1, MPI_DOUBLE, ((op == OP_MIN) ? MPI_MIN : MPI_SUM), MPI_COMM_WORLD) == MPI_SUCCESS) { //Single double operand
            args.rval().setNumber(result);
            return true;
        }
        else {
            args.rval().setUndefined();
            JS_ReportError(cx, "%s: MPI_Allreduce (Op Type: %s) was unsuccessful", __FUNCTION__, ((op == OP_MIN) ? "MPI_MIN" : "MPI_SUM"));
            return false;
        }
    }
}

/*
function MPI.alltoallSum(self)
    if (MPI_Alltoall(self.sndCounts, 1, self.LONG,
            self.rcvCounts, 1, self.LONG, self.comm) ~= SUCCESS) then
        error("Could not AllToAll in MPI")
    end

    local toRcv = 0
    for i=0,self.numRanks-1 do
        toRcv = toRcv + self.rcvCounts[i]
        self.sndCounts[i] = 0
    end

    return toRcv
end

function MPI.sendAndCount(self, x, dst, tag) --Blocking
    local m = msg.pack(x)
    local tag = tag or #m --Set to message length if nil
    if MPI_Send(m, #m, self.BYTE, dst, tag, self.comm) ~= SUCCESS then
        error("Could not Send in MPI")
    end
    self.sndCounts[dst] = self.sndCounts[dst] + 1
end
*/

static bool
MPIAlltoallSum(JSContext* cx, unsigned argc, Value* vp) {
    CallArgs args = CallArgsFromVp(argc, vp);

    if (MPI_Alltoall(Simian_sndCounts, 1, MPI_INT,
            Simian_rcvCounts, 1, MPI_INT, MPI_COMM_WORLD) != MPI_SUCCESS) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: MPI_Alltoall was unsuccessful", __FUNCTION__);
        return false;
    }

    int toRcv = 0;
    for (int i=0; i < Simian_numRanks; i++) {
        toRcv += Simian_rcvCounts[i];
        Simian_sndCounts[i] = 0;
    }

    args.rval().setInt32(toRcv);
    return true;
}

static bool
MPISendAndCount(JSContext* cx, unsigned argc, Value* vp) { //Blocking //IN: ArrayBuffer message, int dst, int tag[null==msgLength], OUT: boolean/undefined status
    CallArgs args = CallArgsFromVp(argc, vp);

    if ((args.length() < 3) || !(args[0].isObject()) || !(args[1].isNumber()) || !(args[2].isNumber() || args[2].isNull())) {
        args.rval().setUndefined();
        JS_ReportError(cx, "%s: Wrong number of arguments or arg[0] (message) is not an Object or arg[1] (dst) is not Int32 or arg[2] (tag) is not Int32/Null", __FUNCTION__);
        return false;
    }
    else {
        AutoCheckCannotGC nogc;
        RootedObject messageObj(cx, &args[0].toObject());
        if (JS_IsArrayBufferObject(messageObj)) {
            uint32_t length = JS_GetArrayBufferByteLength(messageObj);
            void *buffer = JS_StealArrayBufferContents(cx, messageObj);
            int32_t dst = (int32_t)(args[1].toNumber());
            int32_t tag = args[2].isNull() ? length : (int32_t)(args[2].toNumber());

            bool retVal;
            if (MPI_Send((void *)buffer, length, MPI_BYTE, dst, tag, MPI_COMM_WORLD) == MPI_SUCCESS) {
                args.rval().setBoolean(true);
                retVal = true;
                Simian_sndCounts[dst]++;
            }
            else {
                args.rval().setBoolean(false);
                JS_ReportError(cx, "%s: MPI_Send was unsuccessful", __FUNCTION__);
                retVal = false;
            }
            free(buffer);
            return retVal;
        }
        else {
            args.rval().setUndefined();
            JS_ReportError(cx, "%s: arg[0] (message) was not an ArrayBuffer Object", __FUNCTION__);
            return false;
        }
    }
}
