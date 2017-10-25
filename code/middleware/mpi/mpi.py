#
# mpi.py :- a simple mpi model
#

import math, copy
from collections import deque
from interconnect import *

def mpi_comm_rank(mpi_comm):
    """Returns the mpi rank"""
    return mpi_comm['rank']

def mpi_comm_size(mpi_comm):
    """Returns the number of mpi processes in the communicator."""
    return len(mpi_comm['hostmap'])

def mpi_finalize(mpi_comm):
    """It is required all mpi processes end with this function."""
    mpi_comm['mpiproc'].hibernate()

def mpi_send(to_rank, data, sz, mpi_comm, type="default"):
    """Sends an mpi message.

    This function sends user data 'data' of given size 'sz' (in bytes)
    and message type to another mpi process 'to_rank' (it's ok to send
    data to itself). This function returns a boolean indicating
    whether the send operation has been successful. Data could be
    None. Note that the user should not use a message type that starts
    with '__'; it's reserved.
    """

    if not (0 <= to_rank < len(mpi_comm['hostmap'])):
        raise Exception("mpi_send to_rank (%d) out of range (comm=%d, size=%d)" %
                        (to_rank, mpi_comm['commid'], len(mpi_comm['hostmap'])))

    host = mpi_comm['host']
    if "mpi" in host.hpcsim_dict["debug_options"]:
        print("rank %d (comm=%d) mpi_send: to_rank=%d, sz=%d, type=%s" % 
              (mpi_comm['rank'], mpi_comm['commid'], to_rank, sz, type))

    proc = mpi_comm['mpiproc']
    if host.mpi_call_time > 0: proc.sleep(host.mpi_call_time)

    from_true_rank = get_mpi_true_rank(mpi_comm, mpi_comm['rank'])
    to_true_rank = get_mpi_true_rank(mpi_comm, to_rank)
    if from_true_rank not in host.send_msgid:
        host.send_msgid[from_true_rank] = dict()
    if to_true_rank not in host.send_msgid[from_true_rank]:
        host.send_msgid[from_true_rank][to_true_rank] = dict()
    if type not in host.send_msgid[from_true_rank][to_true_rank]:
        host.send_msgid[from_true_rank][to_true_rank][type] = 0
    msg_id = host.send_msgid[from_true_rank][to_true_rank][type]
    host.send_msgid[from_true_rank][to_true_rank][type] += 1
    to_host = get_mpi_true_host(mpi_comm, to_rank)

    # we may have to break data down to pieces according to the min
    # and max packet size requirement (we do this only if source and
    # destination are not on the same host)
    if host.node_id == to_host:
        num_pieces = 1
    else:
        num_pieces = int((sz+host.mpi_maxsz-1)/host.mpi_maxsz)
    piece_idx = 0

    # determine using GET or PUT
    if sz <= host.mpi_putget_thresh: is_get = False
    else: is_get = True
    
    # this is a mechanism to stranggle the send rate
    outstanding = 0
    outstanding_bytes = 0

    while sz > 0:
        # if source and destination are not on the same host, we break
        # them; otherwise, we have one big piece
        if sz <= host.mpi_maxsz or host.node_id == to_host:
            sendsz = sz
        else:
            sendsz = host.mpi_maxsz
        sz -= sendsz
        senditem = {
            "to_rank" : to_rank,
            "to_true_rank" : to_true_rank,
            "to_host" : to_host,
            "from_rank" : mpi_comm['rank'],
            "comm_id" : mpi_comm['commid'],
            "msg_id" : msg_id, # so that we don't confuse successive messages from same rank
            "padded_size" : host.mpi_minsz if sendsz<host.mpi_minsz else sendsz,
            "data_size" : sendsz, # without padding
            "data" : data if piece_idx == 0 else None, # data only attach data to first piece
            "type" : type,
            "piece_idx" : piece_idx,
            "num_pieces" : num_pieces,
            "mpi_process" : proc,
            "data_overhead" : host.mpi_get_datahdr if is_get else host.mpi_put_datahdr,
            "ack_overhead" : host.mpi_get_ackhdr if is_get else host.mpi_put_ackhdr,
        }
        piece_idx += 1
        host.send_buffer.append(senditem)
        if len(host.send_buffer) == 1:
            host.wakeProcess("send_process")
            
        # control send rate only if not on the same host
        outstanding += 1
        outstanding_bytes += sendsz
        if outstanding_bytes >= host.mpi_bufsz and \
           host.node_id != to_host:
            succ = proc.hibernate()
            if not succ: return False
            else: outstanding -= 1

    while 0 < outstanding:
        succ = proc.hibernate()
        if not succ: return False
        else: outstanding -= 1
        
    if "mpi" in host.hpcsim_dict["debug_options"]:
        print("rank %d (comm=%d) mpi_send: to_rank=%d, type=%s: DONE" % 
              (mpi_comm['rank'], mpi_comm['commid'], to_rank, type))
    return True

def mpi_isend(to_rank, data, sz, mpi_comm, type="default"):
    """Immediate mpi send.

    Immediate send function returns immediately (without waiting for
    the message to be sent and acknowledged). This function returns an
    mpi request handle, which the user is expected to use to check the
    completion of the operation (using mpi_wait or mpi_test).
    """

    if not (0 <= to_rank < len(mpi_comm['hostmap'])):
        raise Exception("mpi_isend to_rank (%d) out of range (comm=%d, size=%d)" %
                        (to_rank, mpi_comm['commid'], len(mpi_comm['hostmap'])))

    host = mpi_comm['host']
    if "mpi" in host.hpcsim_dict["debug_options"]:
        print("rank %d (comm=%d) mpi_isend: to_rank=%d, sz=%d, type=%s" % 
              (mpi_comm['rank'], mpi_comm['commid'], to_rank, sz, type))

    proc = mpi_comm['mpiproc']
    if host.mpi_call_time > 0: proc.sleep(host.mpi_call_time)

    from_true_rank = get_mpi_true_rank(mpi_comm, mpi_comm['rank'])
    to_true_rank = get_mpi_true_rank(mpi_comm, to_rank)
    if from_true_rank not in host.send_msgid:
        host.send_msgid[from_true_rank] = dict()
    if to_true_rank not in host.send_msgid[from_true_rank]:
        host.send_msgid[from_true_rank][to_true_rank] = dict()
    if type not in host.send_msgid[from_true_rank][to_true_rank]:
        host.send_msgid[from_true_rank][to_true_rank][type] = 0
    msg_id = host.send_msgid[from_true_rank][to_true_rank][type]
    host.send_msgid[from_true_rank][to_true_rank][type] += 1


    to_host = get_mpi_true_host(mpi_comm, to_rank)

    # we may have to break data down to pieces according to the min
    # and max packet size requirement (we do this only if source and
    # destination are not on the same host)
    if host.node_id == to_host:
        num_pieces = 1
    else:
        num_pieces = int((sz+host.mpi_maxsz-1)/host.mpi_maxsz)

    # a send request is basically used to keep track of whether all
    # pieces have been successfully sent and acknowledged
    sendreq = {
        "mpi_comm" : mpi_comm,
        "status" : "sending", # three possible states: sending, success, or failed
        "pieces_left" : set(range(num_pieces))
    }

    # IMPORTANT: for isend, we don't rate limit for simplicity; we
    # assume it is never going to breach the send rate limit!!!

    piece_idx = 0
    while sz > 0:
        sendsz = sz if sz <= host.mpi_maxsz else host.mpi_maxsz
        # if source and destination are not on the same host, we break
        # them; otherwise, we have one big piece
        if sz <= host.mpi_maxsz or host.node_id == to_host:
            sendsz = sz
        else:
            sendsz = host.mpi_maxsz
        sz -= sendsz
        senditem = {
            "to_rank" : to_rank,
            "to_true_rank" : to_true_rank,
            "to_host" : to_host,
            "from_rank" : mpi_comm['rank'],
            "comm_id" : mpi_comm['commid'],
            "msg_id" : msg_id, # so that we don't confuse successive messages from same rank
            "padded_size" : host.mpi_minsz if sendsz<host.mpi_minsz else sendsz,
            "data_size" : sendsz, # without padding
            "data" : data if piece_idx == 0 else None, # data only attach data to first piece
            "type" : type,
            "piece_idx" : piece_idx,
            "num_pieces" : num_pieces,
            "mpi_request" : sendreq,
            
            #TODO: check it...
            "data_overhead" : host.mpi_get_datahdr, 
            "ack_overhead" : host.mpi_get_ackhdr,
        }
        piece_idx += 1
        host.send_buffer.append(senditem)
        if len(host.send_buffer) == 1:
            host.wakeProcess("send_process")

    return sendreq

def mpi_recv(mpi_comm, from_rank=None, type=None):
    """Blocking mpi receive.

    This function receives user data sent from given rank (or from any
    if from_rank=None) and of give message type (or any type if
    type=None). The function returns None if it fails. Otherwise, it
    returns a dictionary containing from_rank, type, data, and
    data_size. Note that the user should not use type that starts with
    '__'; it's reserved.
    """

    if from_rank is not None and \
       not (0 <= from_rank < len(mpi_comm['hostmap'])):
        raise Exception("mpi_recv from rank (%d) out of range (comm=%d, size=%d)" %
                        (from_rank, mpi_comm['commid'], len(mpi_comm['hostmap'])))

    host = mpi_comm['host']
    if "mpi" in host.hpcsim_dict["debug_options"]:
        print("rank %d (comm=%d) mpi_recv: from_rank=%r, type=%r" % 
              (mpi_comm['rank'], mpi_comm['commid'], from_rank, type))

    proc = mpi_comm['mpiproc']
    if host.mpi_call_time > 0: proc.sleep(host.mpi_call_time)

    to_rank = mpi_comm['rank']
    recvitem = host.check_recv_buffer(mpi_comm, to_rank, from_rank, type, proc, None)
    if recvitem is None:
        recvitem = proc.hibernate()
        if not recvitem: return None

    if "mpi" in host.hpcsim_dict["debug_options"]:
        print("rank %d (comm=%d) mpi_recv: from_rank=%r, type=%r: DONE" % 
              (mpi_comm['rank'], mpi_comm['commid'], recvitem["from_rank"], recvitem["type"]))
    return {
        "from_rank" : recvitem["from_rank"],
        "type" : recvitem["type"],
        "data" : recvitem["data"],
        "data_size" : recvitem["data_size"]
    }

def mpi_irecv(mpi_comm, from_rank=None, type=None):
    """Immediate mpi receive.

    Immediate recv returns immediately (without waiting for the
    message to be received). the function is similar to the blocking
    receive 'mpi_recv', except that it returns an mpi request handle
    which the user is expected to use to check the completion of the
    operation (using mpi_wait or mpi_test).
    """

    if from_rank is not None and \
       not (0 <= from_rank < len(mpi_comm['hostmap'])):
        raise Exception("mpi_recv from rank (%d) out of range (comm=%d, size=%d)" %
                        (from_rank, mpi_comm['commid'], len(mpi_comm['hostmap'])))

    host = mpi_comm['host']
    if "mpi" in host.hpcsim_dict["debug_options"]:
        print("rank %d (comm=%d) mpi_irecv: from_rank=%r, type=%r" % 
              (mpi_comm['rank'], mpi_comm['commid'], from_rank, type))

    proc = mpi_comm['mpiproc']
    if host.mpi_call_time > 0: proc.sleep(host.mpi_call_time)

    to_rank = mpi_comm['rank']
    recvreq = {
        "mpi_comm" : mpi_comm,
        "status" : "receiving", # three possible states: receiving, success, or failed
    }
    host.check_recv_buffer(mpi_comm, to_rank, from_rank, type, None, recvreq)
    return recvreq

def mpi_wait(req):
    """Waits for a send/recv request to complete.

    Specifically for irecv, this function returns a dictionary
    containing information about the received message, including
    from_rank, data, data_size, and type.
    """

    host = req["mpi_comm"]["host"]
    if "mpi" in host.hpcsim_dict["debug_options"]:
        print("rank %d (comm=%d) mpi_wait" % 
              (req["mpi_comm"]['rank'], req["mpi_comm"]['commid']))

    if req["status"] != "success" and req["status"] != "failed":
        proc = req["mpi_comm"]["mpiproc"]
        req["mpi_process"] = proc
        #print("rank %d (comm=%d) mpi_wait hibernate (req=%r)" % 
        #      (req["mpi_comm"]['rank'], req["mpi_comm"]['commid'], req))
        proc.hibernate()
        #print("rank %d (comm=%d) mpi_wait wakeup (req=%r)" % 
        #      (req["mpi_comm"]['rank'], req["mpi_comm"]['commid'], req))

    if req["status"] == "success":
        if "from_rank" in req:
            if "mpi" in host.hpcsim_dict["debug_options"]:
                print("rank %d (comm=%d) mpi_wait: DONE irecv: from_rank=%r, type=%r" % 
                      (req["mpi_comm"]['rank'], req["mpi_comm"]['commid'], 
                       req["from_rank"], req["type"]))
            return {
                "from_rank" : req["from_rank"],
                "type" : req["type"],
                "data" : req["data"],
                "data_size" : req["data_size"]
            } # success with irecv
        else:
            if "mpi" in host.hpcsim_dict["debug_options"]:
                print("rank %d (comm=%d) mpi_wait: DONE isend" % 
                      (req["mpi_comm"]['rank'], req["mpi_comm"]['commid']))
            return None # success with isend
    else:
        return None # failed isend (or irecv)

def mpi_waitall(reqs):
    """Waits for a list of requests.

    The difference from 'wait' is that the function does not return
    anything. In particular, we don't return the received packet info
    for an immediate receive request. The user can get that from the
    corresponding request handle when this function returns 9until all
    requests are completed).
    """
    for r in reqs: mpi_wait(r)

def mpi_test(req):
    """Tests whether an immediate operation has completed.

    This function returns true or false, indicating whether the
    operation has completed (regardless whether successful or
    failed). Note that if the request has completed, for receive, the
    request handle itself already contains information about the
    received message: from_rank, data, data_size, and type.
    """
    return req["status"] == "success" or req["status"] == "failed"

def mpi_sendrecv(to_rank, data, sz, from_rank, mpi_comm, 
                 send_type="default", recv_type="default"):
    """Combines mpi send and receiv in one function.

    Actually, the function first sends 'data' and then receives; it
    returns the same as mpi_recv (i.e., the receive information is
    returned as a dictionary. Either from_rank or to_rank can be None,
    in which case the corresponding receive or send operation is
    bypassed. Note that the user should not use type that starts with
    '__'; it's reserved.
    """

    if to_rank is not None:
        mpi_send(to_rank, data, sz, mpi_comm, type=send_type)
    if from_rank is not None:
        return mpi_recv(mpi_comm, from_rank, type=recv_type)

def mpi_reduce(root, data, mpi_comm, data_size=4, op="sum"):
    """MPI reduction.

    Available reduction operators include "sum", "prod", "min", and
    "max".  As expected, only root rank gets the reduction result.
    """

    n = len(mpi_comm['hostmap'])
    p = mpi_comm['rank']
    if not (0 <= root < n):
        raise Exception("mpi_reduce root (%d) out of range (comm=%d, size=%d)" % 
                        (root, mpi_comm['commid'], n))

    # one node situation
    if n <= 1: return data

    # find out the number of binomial tree steps
    steps = int(math.ceil(math.log(n,2)))

    # my new rank (swapping 0 and root)
    if p == root: rank = 0
    elif p == 0: rank = root
    else: rank = p

    mid = int(pow(2, steps-1))
    for s in range(steps):
        if rank >= mid:
            # send in this round and quit when done
            t = rank - mid
            if t == root: t = 0
            elif t == 0: t = root
            #print("reduce: [[[%d]]] sends to %d with d=%d" % (p, t, data))
            r = mpi_send(t, data, data_size, mpi_comm, type="__reduce__")
            #print("reduce: [[[%d]]] done reduce" % p)
            if r is None: return None
            else: return data
        else:
            # receive in this round, but only if partner exists
            if rank + mid < n:
                #print("reduce: [[[%d]]] expecting to receive" % p)
                x = mpi_recv(mpi_comm, type="__reduce__")
                if x is None: return None
                #print("reduce: [[[%d]]] receives from %d got d=%d (sum: %d)" % 
                #      (p, x["from_rank"], x["data"], data+x["data"]))

                if op == 'sum': data += x["data"]
                elif op == 'prod': data *= x["data"]
                elif op == 'max': data = data if data > x["data"] else x["data"]
                elif op == 'min': data = data if data < x["data"] else x["data"]
                else: raise Exception("reduce operator %s not implemented" % op)
                    
            # a sender would carry on to the next round
            mid /= 2
    else: return data

def mpi_gather(root, data, mpi_comm, data_size=4):
    """Returns a vector of values gathered from all processes."""

    n = len(mpi_comm['hostmap'])
    p = mpi_comm['rank']
    if not (0 <= root < n):
        raise Exception("mpi_gather root (%d) out of range (comm=%d, size=%d)" % 
                        (root, mpi_comm['commid'], n))

    res_dict = dict()
    res_dict[p] = data

    # one node situation
    if n <= 1: 
        result = dict([(i,0) for i in range(n)])
        result.update(res_dict)
        return [result[k] for k in sorted(result.keys())]

    # find out the number of binomial tree steps
    steps = int(math.ceil(math.log(n,2)))

    # my new rank (swapping 0 and root)
    if p == root: rank = 0
    elif p == 0: rank = root
    else: rank = p

    mid = int(pow(2, steps-1))
    for s in range(steps):
        if rank >= mid:
            # send in this round and quit when done
            t = rank - mid
            if t == root: t = 0
            elif t == 0: t = root
            #print("reduce: [[[%d]]] sends to %d with d=%d" % (p, t, data))
            r = mpi_send(t, res_dict, data_size*(s+1), mpi_comm, type="__gather__")
            #print("reduce: [[[%d]]] done reduce" % p)
            if r is None: return None
            result = dict([(i,0) for i in range(n)])
            result.update(res_dict)
            return [result[k] for k in sorted(result.keys())]
        else:
            # receive in this round, but only if partner exists
            if rank + mid < n:
                #print("reduce: [[[%d]]] expecting to receive" % p)
                x = mpi_recv(mpi_comm, type="__gather__")
                if x is None: return None
                #print("reduce: [[[%d]]] receives from %d got d=%d (sum: %d)" % 
                #      (p, x["from_rank"], x["data"], data+x["data"]))
                res_dict.update(x["data"])
                    
            # a sender would carry on to the next round
            mid /= 2
    else: return [res_dict[k] for k in sorted(res_dict.keys())]

def mpi_allgather(data, mpi_comm, data_size=4):
    """All gather.

    It is implemented by first gathering at rank 0 then bcast.
    """

    r = mpi_gather(0, data, mpi_comm, data_size)
    if r is None: return None
    else: return mpi_bcast(0, r, mpi_comm, data_size*len(mpi_comm['hostmap']))

def mpi_bcast(root, data, mpi_comm, data_size=4):
    """Returns the data provided only by the root."""

    n = len(mpi_comm['hostmap'])
    p = mpi_comm['rank']
    if not (0 <= root < n):
        raise Exception("mpi_bcast root (%d) out of range (comm=%d, size=%d)" % 
                        (root, mpi_comm['commid'], n))

    # one node situation
    if n <= 1: return data

    # find out the number of binomial tree steps
    steps = int(math.ceil(math.log(n,2)))

    # my new rank (swapping 0 and root)
    if p == root: rank = 0
    elif p == 0: rank = root
    else: rank = p

    mid = 1
    for s in range(steps):
        # skip the round if it's not my turn
        if rank < 2*mid: 
            if rank >= mid:
                # receive in this round
                #print("bcast: [[[%d]]] expecting to receive" % p)
                r = mpi_recv(mpi_comm, type="__bcast__")
                if r is None: return None
                #print("bcast: [[[%d]]] receives from %d got d=%d" % 
                #      (p, r["from_rank"], r["data"]))
                data = r['data']
            else:
                # send in this round, but only if partner exists
                t = rank + mid
                if t < n: 
                    if t == root: t = 0
                    elif t == 0: t = root
                    #print("bcast: [[[%d]]] sends to %d with d=%d" % (p, t, data))
                    r = mpi_send(t, data, data_size, mpi_comm, type="__bcast__")
                    if r is None: return None
            
        # carry on to the next round
        mid *= 2
    else: return data

def mpi_scatter(root, data, mpi_comm, data_size=4):
    """Takes a vector from root and distribute the elements to all ranks."""

    n = len(mpi_comm['hostmap'])
    p = mpi_comm['rank']
    if not (0 <= root < n):
        raise Exception("mpi_bcast root (%d) out of range (comm=%d, size=%d)" % 
                        (root, mpi_comm['commid'], n))

    # it'd better be the data is a list of n elements
    res_dict = dict([(i,data[i]) for i in range(n)])

    # one node situation
    if n <= 1: return res_dict[p]

    # find out the number of binomial tree steps
    steps = int(math.ceil(math.log(n,2)))

    # my new rank (swapping 0 and root)
    if p == root: rank = 0
    elif p == 0: rank = root
    else: rank = p

    mid = 1
    for s in range(steps):
        # skip the round if it's not my turn
        if rank < 2*mid: 
            if rank >= mid:
                # receive in this round
                #print("bcast: [[[%d]]] expecting to receive" % p)
                r = mpi_recv(mpi_comm, type="__scatter__")
                if r is None: return None
                #print("bcast: [[[%d]]] receives from %d got d=%d" % 
                #      (p, r["from_rank"], r["data"]))
                res_dict = r['data']
            else:
                # send in this round, but only if partner exists
                t = rank + mid
                if t < n: 
                    if t == root: t = 0
                    elif t == 0: t = root
                    #print("bcast: [[[%d]]] sends to %d with d=%d" % (p, t, data))
                    r = mpi_send(t, res_dict, data_size*n/(mid*2), mpi_comm, type="__scatter__")
                    if r is None: return None
            
        # carry on to the next round
        mid *= 2
    else: return res_dict[p]

def mpi_barrier(mpi_comm):
    """Barrier is implemented as reduce and then bcast."""

    mpi_reduce(0, 0, mpi_comm)
    mpi_bcast(0, 0, mpi_comm)

def mpi_allreduce(data, mpi_comm, data_size=4, op="sum"):
    """Implemented as reduce and then bcast."""

    #print("%d before reduce" % mpi_comm['rank'])
    data = mpi_reduce(0, data, mpi_comm, data_size, op)
    #print("%d between reduce and bcast" % mpi_comm['rank'])
    if data is None: return None
    else: return mpi_bcast(0, data, mpi_comm, data_size)
        #print("%d after bcast" % mpi_comm['rank'])

def mpi_alltoall(data, mpi_comm, data_size=4):
    """All-to-all is a matrix transpose operation.

    Each rank sends a different message to other ranks. Data must be a
    list of n elements, where n is the total of processes; data_size
    is the size of data to be sent to each of the processes.
    """

    n = len(mpi_comm['hostmap'])
    p = mpi_comm['rank']

    steps = int(math.ceil(math.log(n,2)))
    if n == int(math.pow(2, steps)):
        # if power of two, we use hypercube pairwise exchange method
        matrix = dict()
        matrix[p] = data

        m = 1
        for s in range(steps):
            q = p ^ m
            # we simply gather data from all processes rather than
            # trying to figure out exactly what needs to be exchanged
            r = mpi_sendrecv(q, matrix, data_size*n/2, q, mpi_comm, 
                             send_type = "__alltoall__", recv_type = "__alltoall__")
            if r is None: return None
            matrix.update(r['data'])
            m *= 2
        res = []
        for i in range(n):
            res.append(matrix[i][p])
        return res
    else:
        # otherwise, use point to point exchange method
        for i in range(n):
            for j in range(i, n):
                if p  == i:
                    r = mpi_sendrecv(j, data[j], data_size, j, mpi_comm, 
                                     send_type = "__alltoall__", recv_type = "__alltoall__")
                    if r is None: return None
                    data[j] = r['data']
                elif p == j:
                    r = mpi_sendrecv(i, data[i], data_size, i, mpi_comm, 
                                     send_type = "__alltoall__", recv_type = "__alltoall__")
                    if r is None: return None
                    data[i] = r['data']
        return data

def mpi_alltoallv(data, mpi_comm, data_sizes=None):
    """All-to-all with variable data size.

    The difference of alltoallv and alltoall is that the former needs
    a list for the data sizes (one element for each mpi process); if
    not provided, we assume everything is 4 bytes.
    """

    n = len(mpi_comm['hostmap'])
    p = mpi_comm['rank']

    if data_sizes is None:
        data_sizes = [4]*n

    # we only use point to point exchange method
    for i in range(n):
        for j in range(i, n):
            if p  == i:
                r = mpi_sendrecv(j, data[j], data_sizes[j], j, mpi_comm, 
                                 send_type = "__alltoall__", recv_type = "__alltoall__")
                if r is None: return None
                data[j] = r['data']
            elif p == j:
                r = mpi_sendrecv(i, data[i], data_sizes[i], i, mpi_comm, 
                                 send_type = "__alltoall__", recv_type = "__alltoall__")
                if r is None: return None
                data[i] = r['data']
    return data

def mpi_comm_split(mpi_comm, color, key):
    """Split the mpi communicator.

    The function returns a new communicator: processes with the same
    color (in mpi it's an integer, here it can be any type) are in the
    same new communicator. If color is "mpi_undefined" (case
    insensitive here), the process is in its own single member group.
    The key controls the rank assignment in the new communicator.
    """

    # 1. use mpi_allgather to get the color and key from each process
    r = mpi_allgather((color, key, mpi_comm['rank']), mpi_comm, 8)
    if r is None: 
        raise Exception("mpi_comm_split: allgather failed")
        return None

    if (type(color) is str) and (color.lower() == 'mpi_undefined'):
        # 2. special treatment: if the color is "mpi_undefined",
        # returns mpi_comm_null (a group with a single member)

        # regardless whether it's used, we allocate one (even if all
        # colors are mpi_undefined)
        alloc_new_mpi_comm(mpi_comm, None)

        mpi_comm_world = get_mpi_comm_ancestor(mpi_comm)
        return mpi_comm_world['comms'][1]

    else:
        # 3. for processes with the same color, create a communicator
        # with that many processes; use key to order the ranks

        # create new communicator structure (with parent_comm pointing
        # to the parent communicator given in the argument)
        new_comm = dict()
        new_comm["host"] = mpi_comm['host']
        new_comm["mpiproc"] = mpi_comm["mpiproc"]
        new_comm["parent_comm"] = mpi_comm
        new_comm["hostmap"] = list()
        r.sort(key = lambda t: t[1]) # sort with increasing key
        for i, j, k in r:
            if color == i:
                if k == mpi_comm["rank"]:
                    new_comm["rank"] = len(new_comm["hostmap"])
                new_comm["hostmap"].append(k)
        #assert 'rank' in new_comm
        alloc_new_mpi_comm(mpi_comm, new_comm)
        return new_comm

def mpi_comm_dup(mpi_comm):
    """Duplicates the communicator."""

    new_comm = copy.copy(mpi_comm)
    alloc_new_mpi_comm(mpi_comm, new_comm)
    return new_comm

def mpi_comm_free(mpi_comm):
    """Reclaims the communicator."""

    # it is possible that other processes are still using the
    # communicator; we do a barrier to avoid that, instead of
    # reclaiming the communicator slot right away; this means that
    # mpi_comm_free is a collective call
    mpi_barrier(mpi_comm)

    # you can't remove mpi_comm_world and mpi_comm_null
    if mpi_comm['commid'] > 2:
        mpi_comm_world = get_mpi_comm_ancestor(mpi_comm)
        del mpi_comm_world['comms'][mpi_comm['commid']]
    
def mpi_comm_group(mpi_comm):
    """Returns the group corresponding to the given communicator."""

    # the set of processes is represented by hostmap; a group must be
    # related to a communicator
    return { 
        'group' : range(len(mpi_comm['hostmap'])),
        'comm' : mpi_comm
    }

def mpi_group_size(mpi_group):
    """Returns the size of a group."""
    return len(mpi_group['group'])

def mpi_group_rank(mpi_group):
    """Returns the rank of this process in the given group.

    The function returns None if the process is not part of the group.
    """

    r = mpi_group['comm']['rank']
    for idx in xrange(len(mpi_group['group'])):
        if r == mpi_group['group'][idx]:
            return idx

def mpi_group_free(mpi_group):
    """Reclaims the group."""
    # we need not do anything
    pass

def mpi_group_incl(mpi_group, ranks):
    """Returns a new group by including ranks listed in order."""

    g = list()
    for r in ranks:
        if (r in mpi_group['group']) and (r not in g):
            g.append(r)
    return {
        'group' : g,
        'comm' : mpi_group['comm']
    }

def mpi_group_excl(mpi_group, ranks):
    """Returns a new group by excluding ranks listed.

    The new group is created by reordering an existing group and
    taking only unlisted members, preserving the order defined by
    orginal group.
    """
    
    g = copy.copy(mpi_group['group'])
    for r in ranks:
        if r in g: g.remove(r)
    return {
        'group' : g,
        'comm' : mpi_group['comm']
    }

def mpi_comm_create(mpi_comm, mpi_group):
    """Creates and returns a new communicator from group.

    If this process is not in the group, it'll be in its own single
    member group.
    """

    if mpi_comm != mpi_group['comm']:
        raise Exception("mpi_comm_create: group does not belong to the communicator")

    r = mpi_group_rank(mpi_group)
    if r is None:
        # regardless whether it's used, we allocate one
        alloc_new_mpi_comm(mpi_comm, None)
        mpi_comm_world = get_mpi_comm_ancestor(mpi_comm)
        return mpi_comm_world['comms'][1]
    else:
        new_comm = dict()
        new_comm["host"] = mpi_comm['host']
        new_comm["mpiproc"] = mpi_comm["mpiproc"]
        new_comm["parent_comm"] = mpi_comm
        new_comm["hostmap"] = mpi_group['group']
        new_comm["rank"] = r
        alloc_new_mpi_comm(mpi_comm, new_comm)
        return new_comm

def mpi_comm_create_group(mpi_comm, mpi_group, tag=None):
    """Creates and returns a new communicator from group.

    We make this function the same as the previous, the tag is not
    used.
    """
    return mpi_comm_create(mpi_comm, mpi_group)

def mpi_cart_create(mpi_comm, dims, periodic=None):
    """Returns a communicator added with cartesian coordinates.

    If periodic is None, we assume it's true for all
    dimensions. Otherwise, it is expected to be a vector of booleans
    having the same cardinality as 'dims'. Note that reorder is not
    used and nor is it included here as an argument.
    """

    if periodic is not None and len(dims) != len(periodic):
        raise Exception("mpi_cart_create: unmatched dims %r and periodic %r" %
                        (dims, periodic))
    # we require dimensions should match with the comm size
    total = dims[0]
    for d in dims[1:]: total *= d
    if len(mpi_comm['hostmap']) != total:
        raise Exception("mpi_cart_create: unmatched dims %r with comm_size %d" %
                        (dims, len(mpi_comm['hostmap'])))

    new_comm = mpi_comm_dup(mpi_comm)
    new_comm['cart'] = dict()
    new_comm['cart']['dims'] = copy.copy(dims)
    if periodic is None:
        new_comm['cart']['periodic'] = (True,)*len(dims)
    else:
        new_comm['cart']['periodic'] = copy.copy(periodic)
    new_comm['cart']['cm'] = [1]*len(dims)
    for d in xrange(len(dims)-2, -1, -1): # dimension in reverse order
        new_comm['cart']['cm'][d] = new_comm['cart']['cm'][d+1]*dims[d+1]
    return new_comm

def mpi_cart_coords(mpi_comm, rank):
    """Returns the cartesian coordinates of the given rank."""

    if 'cart' not in mpi_comm:
        raise Exception("mpi_cart_coords: communicator not cartesian")
    if not (0 <= rank < len(mpi_comm['hostmap'])):
        raise Exception("mpi_cart_coords: invalid rank=%d" % rank)
    
    c = []
    for d in xrange(len(mpi_comm['cart']['dims'])):
        c.append(rank/mpi_comm['cart']['cm'][d])
        rank %= mpi_comm['cart']['cm'][d]
    return tuple(c)

def mpi_cart_rank(mpi_comm, coords):
    """Returns the rank of given cartesian coordinates."""

    if 'cart' not in mpi_comm:
        raise Exception("mpi_cart_rank: communicator not cartesian")
    if len(coords) != len(mpi_comm['cart']['dims']):
        raise Exception("mpi_cart_rank: invalid coords %r" % coords)
    r = 0
    for d in xrange(len(coords)):
        if not (0 <= coords[d] < mpi_comm['cart']['dims'][d]):
            raise Exception("mpi_cart_rank: invalid coords %r" % coords)
        r += coords[d]*mpi_comm['cart']['cm'][d]
    return r

def mpi_cart_shift(mpi_comm, shiftdim, disp):
    """Returns the source and destination ranks as a tuple.

    'shftdim' is the coordinate dimension to shift; 'disp' is the
    displacement (>0 means upward shift, and <0 means downward shift.
    """

    if 'cart' not in mpi_comm:
        raise Exception("mpi_cart_shift: communicator not cartesian")
    if not (0 <= shiftdim < len(mpi_comm['cart']['dims'])):
        raise Exception("mpi_cart_shift: invalid shiftdim=%d" % shiftdim)

    c = mpi_cart_coords(mpi_comm, mpi_comm['rank'])

    cc = list(c)
    x = cc[shiftdim] + disp
    if 0 <= x < mpi_comm['cart']['dims'][shiftdim]:
        cc[shiftdim] = x
        dst = mpi_cart_rank(mpi_comm, cc)
    elif mpi_comm['cart']['periodic'][shiftdim]: 
        x = x % mpi_comm['cart']['dims'][shiftdim]
        cc[shiftdim] = x
        dst = mpi_cart_rank(mpi_comm, cc)
    else:
        dst = None

    cc = list(c)
    x = cc[shiftdim] - disp
    if 0 <= x < mpi_comm['cart']['dims'][shiftdim]:
        cc[shiftdim] = x
        src = mpi_cart_rank(mpi_comm, cc)
    elif mpi_comm['cart']['periodic'][shiftdim]: 
        x = x % mpi_comm['cart']['dims'][shiftdim]
        cc[shiftdim] = x
        src = mpi_cart_rank(mpi_comm, cc)
    else:
        src = None

    return (src, dst)

def mpi_wtime(mpi_comm):
    """Returns the current time.

    This function is a bit different from the original mpi function
    since we do need mpi_comm as an argument, because the communicator
    contains the process context.
    """
    return mpi_comm['host'].get_now()

def mpi_ext_host(mpi_comm):
    """Returns the host running the mpi process; this is an extension."""
    return mpi_comm['host']

def mpi_ext_sleep(time, mpi_comm):
    """The mpi process sleeps for the given amount of time; this is an extension."""
    mpi_comm['mpiproc'].sleep(time)


def get_mpi_comm_ancestor(mpi_comm):
    """Helper function to get mpi_comm_world.

    Communicators are organized as a tree. mpi_comm_world is the
    ancestor of all other communicators.
    """

    if 'parent_comm' in mpi_comm:
        return get_mpi_comm_ancestor(mpi_comm['parent_comm'])
    else:
        return mpi_comm

def get_mpi_true_rank(mpi_comm, rank):
    """Helper function to get the rank in mpi_comm_world."""

    #print("c=%r, r=%d" % (mpi_comm, rank))
    if "parent_comm" in mpi_comm:
        if not (0 <= rank < len(mpi_comm["hostmap"])):
            raise Exception("ERROR: rank %d out of range (comm=%d)" % 
                            (rank, mpi_comm['commid']))
        return get_mpi_true_rank(mpi_comm["parent_comm"], mpi_comm["hostmap"][rank])
    else: 
        return rank

def get_mpi_true_host(mpi_comm, rank):
    """Helper function to get mapped host of the give rank."""

    if "parent_comm" in mpi_comm:
        if not (0 <= rank < len(mpi_comm["hostmap"])):
            raise Exception("ERROR: rank %d out of range (comm=%d)" % 
                            (rank, mpi_comm['commid']))
        return get_mpi_true_host(mpi_comm["parent_comm"], mpi_comm["hostmap"][rank])
    else: 
        return mpi_comm['hostmap'][rank]

def alloc_new_mpi_comm(mpi_comm, new_comm):
    """Allocates and returns the next unused slot for the new communicator."""

    mpi_comm_world = get_mpi_comm_ancestor(mpi_comm)
    newid = mpi_allreduce(mpi_comm_world['next_commid'], mpi_comm, op="max")
    mpi_comm_world['next_commid'] = newid+1

    if new_comm is not None:
        new_comm['commid'] = newid
        mpi_comm_world['comms'][newid] = new_comm
    #print("%d on %s: allocate comm=%d" % 
    #      (mpi_comm_world['rank'], mpi_comm_world['host'], newid))

    # important to have a barrier here, or some processes may get
    # ahead and send messages using the new communicator to another
    # process, which has not set up yet
    mpi_barrier(mpi_comm)


class MPIHost(Host):
    """A compute node with MPI installed."""

    # local variables: (class derived from Host)
    #   send_buffer: a deque for handing messages to send process
    #   resend_key: the send sequence number
    #   resend_buffer: stores unacknowledged messages (indexed by seqno)
    #   recv_buffer: indexed by receiver's true rank
    #   send_msgid: sequence number from rank to rank
    #   recv_msgid: sequence number from rank to rank
    #   comm_world: everything to do with communicators

    def __init__(self, baseinfo, hpcsim_dict, *args):
        super(MPIHost, self).__init__(baseinfo, hpcsim_dict, *args)
        # args include: intercon, swid, swport, bdw, bufsz, link_delay, mem_bandwidth, mem_bufsiz, mem_delay

        # the process in charge of sending mpi messages, along with
        # the mechanisms to synchronize with the user mpi processes
        #self.send_semaphore = 0
        self.send_buffer = deque()
        self.createProcess("send_process", send_process)
        self.startProcess("send_process")

        # resend buffer is used to store previous sent but unacked
        # messages
        self.resend_key = 0
        self.resend_buffer = dict()

        # receive buffer is divided per receiver's (true) rank
        self.recv_buffer = dict()

        # managing message serial number: 
        # send_msgid[from_true_rank][to_true_rank]
        # recv_msgid[to_true_rank][from_true_rank]
        self.send_msgid = dict()
        self.recv_msgid = dict()

        # managing communicators
        self.comm_world = dict()

    def create_mpi_proc(self, *args):
        """A service scheduled by start_mpi to start user mpi process."""

        data = args[0] 

        # mpi default values are set here
        self.mpi_resend_intv = data["mpiopt"]["resend_intv"]
        self.mpi_resend_trials = data["mpiopt"]["resend_trials"]
        self.mpi_minsz = data["mpiopt"]["min_pktsz"]
        self.mpi_maxsz = data["mpiopt"]["max_pktsz"]
        self.mpi_put_datahdr = data["mpiopt"]["put_data_overhead"]
        self.mpi_put_ackhdr = data["mpiopt"]["put_ack_overhead"]
        self.mpi_get_datahdr = data["mpiopt"]["get_data_overhead"]
        self.mpi_get_ackhdr = data["mpiopt"]["get_ack_overhead"]
        self.mpi_putget_thresh = data["mpiopt"]["putget_thresh"]
        self.mpi_call_time = data["mpiopt"]["call_time"]
        # multiplying the injection rate by round-trip-time (with some
        # slacks, factor of 2) would be the max send window
        self.mpi_bufsz = data["mpiopt"]["max_injection"] * \
                         self.intercon.network_diameter_time()*4

        # create the mpi main function
        proc_name = "%s_%d"% (data["main_proc"], data["rank"])
        self.createProcess(proc_name, kernel_main_function)

        # set up a couple of default communicators: mpi_comm_world and
        # mpi_comm_null
        mpi_comm_world = {
            "host" : self,
            "mpiproc" : self.getProcess(proc_name),
            "hostmap" : data["hostmap"],
            "rank" : data["rank"],
            "commid" : 2,
        }
        mpi_comm_null = {
            "host" : mpi_comm_world['host'],
            "mpiproc" : mpi_comm_world['mpiproc'],
            "parent_comm" : mpi_comm_world,
            "hostmap" : [ mpi_comm_world['rank'] ],
            "rank" : 0,
            "commid" : 1,
        }
        self.comm_world[data["rank"]] = mpi_comm_world
        mpi_comm_world['comms'] = dict()
        mpi_comm_world['comms'][1] = mpi_comm_null
        mpi_comm_world['comms'][2] = mpi_comm_world
        mpi_comm_world['next_commid'] = 3

        # run the mpi main function
        self.startProcess(proc_name, data["main_proc"], mpi_comm_world, *data["args"])

    def send_mpi_message(self, senditem, key):
        """Sends message as a packet with seqno being the resend key."""
    
        #for key in senditem:
        #    print("key: %s, item: %r"%(key, senditem[key]))
        pkt = Packet(self.node_id, # srchost
                     senditem["to_host"], # dsthost
                     'data_mpi', # type
                     key, # seqno is the resend key
                     senditem["padded_size"]+senditem["data_overhead"],
                     nonreturn_data={
                         "to_rank" : senditem["to_rank"],
                         "to_true_rank" : senditem["to_true_rank"],
                         "from_rank" : senditem["from_rank"],
                         "comm_id" : senditem["comm_id"],
                         "msg_id" : senditem["msg_id"],
                         "data_size" : senditem["data_size"],
                         "data" : senditem["data"],
                         "type" : senditem["type"],
                         "piece_idx" : senditem["piece_idx"],
                         "num_pieces" : senditem["num_pieces"],
                         "ack_overhead" : senditem["ack_overhead"],
                     },
                     ttl=self.intercon.network_diameter())
        pkt.set_sendtime(self.get_now())
        pkt.add_to_path(str(self))

        if self.node_id == pkt.dsthost:
            self.mem_queue.send_pkt(pkt)
        else:
            self.interfaces['r'].send_pkt(pkt, 0)

    def resend_mpi_message(self, *args):
        """A service for resending messages upon timeout."""
    
        key = args[0]
        if key in self.resend_buffer:
            self.resend_buffer[key]["trials"] += 1
            if self.resend_buffer[key]["trials"] >= self.mpi_resend_trials:
                # when max resend trials exceeded, send fails; we
                # remove the send item from resend buffer and simply
                # unblock the user process (user is notified of the
                # failure)
                senditem = self.resend_buffer[key]["senditem"]
                print("WARNING: %f: %s max rxmit reached for key=%d, item=%r" % 
                      (self.get_now(), self, key, senditem))
                del self.resend_buffer[key]
                
                assert "mpi_process" in senditem or "mpi_request" in senditem
                if "mpi_process" in senditem:
                    senditem["mpi_process"].wake(False)
                else:
                    senditem["mpi_request"]["status"] = "failed"
                    if "mpi_process" in senditem["mpi_request"]:
                        senditem["mpi_request"]["mpi_process"].wake(False)

            else:
                # send the message and schedule another resend
                print("WARNING: %f: %s retransmit key=%d, retry=%d, item=%r" % 
                      (self.get_now(), self, key, self.resend_buffer[key]["trials"], 
                       self.resend_buffer[key]["senditem"]))
                self.send_mpi_message(self.resend_buffer[key]["senditem"], key)
                self.reqService(self.mpi_resend_intv, "resend_mpi_message", key)
        else:
            # otherwise, the message has already been properly acknowledge
            #print("%f: %s resend wakes for key=%d but finds nothing to do" % 
            #      (self.get_now(), self, key))
            pass

    def notify_ack_recv(self, ack):
        """Override the same method in the host class."""

        if ack.type != 'ack_mpi':
            super(MPIHost, self).notify_ack_recv(ack)
            return

        key = ack.seqno
        if key in self.resend_buffer:
            senditem = self.resend_buffer[key]["senditem"]
            del self.resend_buffer[key]

            assert "mpi_process" in senditem or "mpi_request" in senditem
            if "mpi_process" in senditem:
                senditem["mpi_process"].wake(True)
            else:
                senditem["mpi_request"]["pieces_left"].remove(senditem["piece_idx"])
                if len(senditem["mpi_request"]["pieces_left"]) == 0:
                    senditem["mpi_request"]["status"] = "success"
                    if "mpi_process" in senditem["mpi_request"]:
                        senditem["mpi_request"]["mpi_process"].wake(True)

        # otherwise, the sent message has already been properly acknowledge

    def check_recv_buffer(self, mpi_comm, to_rank, from_rank, type, proc, recvreq):
        """Check whether the given receive request can be immediately satistfied.

        This method is called by mpi_recv and mpi_irecv. If the
        request cannot be satisfied yet, it is entered into the
        receive buffer.
        """

        to_true_rank = get_mpi_true_rank(mpi_comm, to_rank)
        if to_true_rank not in self.recv_buffer:
            self.recv_buffer[to_true_rank] = []
            self.recv_msgid[to_true_rank] = dict()
            recvitem = {
                "from_rank" : from_rank, # possibly None
                "comm_id" : mpi_comm['commid'],
                "type" : type, # possibly None
            }
            if proc: 
                recvitem["mpi_process"] = proc
            else: 
                recvitem["mpi_request"] = recvreq
            self.recv_buffer[to_true_rank].append(recvitem)
            #print("%d: check_recv_buffer<1>: put %r in NEW buffer" % (to_true_rank, recvitem))
            return None
        else:
            # linear search is easy
            #print("%d: check_recv_buffer<2>: try match: comm_id=%d, to_rank=%s, from_rank=%r, type=%s" % \
            #      (to_true_rank, mpi_comm['commid'], to_rank, from_rank, type))
            for idx in range(len(self.recv_buffer[to_true_rank])):
                recvitem = self.recv_buffer[to_true_rank][idx]
                #print("%d: check_recv_buffer<2>: recvitem=%s" % (to_true_rank, recvitem))

                if recvitem['comm_id'] != mpi_comm['commid']: continue

                # if this is a pending request
                if "mpi_process" in recvitem or "mpi_request" in recvitem:
                    # this is a stringent requirement which may cause
                    # problems: one cannot have two outstanding receives
                    # posted with wildcard from-rank and message-type
                    if (recvitem["from_rank"] is None or from_rank is None) and \
                       (recvitem["type"] is None or type is None):
                        raise Exception("ERROR: multiple recv requests to rank %d (comm=%d) found" % 
                                        (to_rank, mpi_comm['commid']))
                    continue

                if from_rank is not None:
                    f = get_mpi_true_rank(mpi_comm, from_rank)
                else:
                    rc = self.comm_world[to_true_rank]['comms'][recvitem["comm_id"]]
                    f = get_mpi_true_rank(rc, recvitem["from_rank"])
                assert f is not None

                t = type
                if type is None: t = recvitem["type"]
                assert t is not None
                
                if f not in self.recv_msgid[to_true_rank]:
                    self.recv_msgid[to_true_rank][f] = dict()
                if t not in self.recv_msgid[to_true_rank][f]:
                    self.recv_msgid[to_true_rank][f][t] = 0
                #print("to_true_rank=%d, f=%d" % (to_true_rank, f))
                #print("expect %d" % self.recv_msgid[to_true_rank][f][t])

                # the match criteria:
                # (1) the request has wildcard from_rank or rank is perfect match, and 
                # (2) type is perfect match, or the reqeust has wildcard type and recv item is not reserved, and
                # (3) the recv item must be the expected one in sequence
                #print("%d: 1=%s,2=%s,3=%s,4=%s,5=%s,6=%s" % \
                #      (to_true_rank, str(from_rank), str(recvitem["from_rank"]),
                #       str(recvitem["type"]), str(type),
                #       str(recvitem.get("msg_id")),
                #       str(self.recv_msgid[to_true_rank][f][t])))
		if (from_rank is None or recvitem["from_rank"] == from_rank) and \
                   (recvitem["type"] == type or type is None and recvitem["type"][:2] != "__") and \
                   ("msg_id" not in recvitem or recvitem["msg_id"] == self.recv_msgid[to_true_rank][f][t]):
                    # if there's a match, we simply go with the first
                    # match even though there might be multiple
                    # matches (say, messages from other ranks), even
                    # if the other matches would result more a
                    # definitive receive at the moment
                    break
            else: recvitem = None

            if recvitem is not None:
                # there would be at most one receive request and
                # recvitem found must be from received data
                assert "mpi_process" not in recvitem and "mpi_request" not in recvitem
                assert "missing_pieces" in recvitem

                if len(recvitem["missing_pieces"]) > 0:
                    # there's still at least one missing piece
                    if proc: 
                        recvitem["mpi_process"] = proc
                    else:
                        recvitem["mpi_request"] = recvreq
                    #print("%d: check_recv_buffer<3>: found %r in buffer, wait for more pieces" % \
                    #      (to_true_rank, recvitem))
                    return None
                else:
                    # we got everything, return it to user
                    del self.recv_buffer[to_true_rank][idx]
                    from_rank = recvitem["from_rank"]
                    rc = self.comm_world[to_true_rank]['comms'][recvitem["comm_id"]]
                    from_true_rank = get_mpi_true_rank(rc, from_rank)
                    t = recvitem["type"]
                    if from_true_rank not in self.recv_msgid[to_true_rank]:
                        self.recv_msgid[to_true_rank][from_true_rank] = dict()
                    if t in self.recv_msgid[to_true_rank][from_true_rank]:
                        self.recv_msgid[to_true_rank][from_true_rank][t] += 1
                    else:
                        self.recv_msgid[to_true_rank][from_true_rank][t] = 1

                    if recvreq:
                        #recvitem["mpi_request"] = recvreq
                        recvreq["status"] = "success"
                        recvreq["from_rank"] = from_rank
                        recvreq["type"] = t
                        recvreq["data"] = recvitem["data"]
                        recvreq["data_size"] = recvitem["data_size"]
                    #print("%d: check_recv_buffer<4>: found %r in buffer" % (to_true_rank, recvitem))
                    return recvitem
            else:
                # there's no match exist, insert the request and
                # return None
                recvitem = {
                    "from_rank" : from_rank,
                    "comm_id" : mpi_comm['commid'],
                    "type" : type,
                }
                if proc: 
                    recvitem["mpi_process"] = proc
                else: 
                    recvitem["mpi_request"] = recvreq
                self.recv_buffer[to_true_rank].append(recvitem)
                #print("%d: check_recv_buffer<5>: put req %r in buffer" % (to_true_rank, recvitem))
                return None

    def notify_data_recv(self, pkt):
        """Overrides the same method in the host class to handle receive.

        This function is called by the receiving process.
        """
        
        if pkt.type != 'data_mpi':
            super(MPIHost, self).notify_data_recv(pkt)
            return

        # generate ack packet
        ack = Packet(self.node_id, pkt.srchost, "ack_mpi", pkt.seqno, 
                     pkt.nonreturn_data["ack_overhead"],
                     return_data=pkt.return_data,
                     ttl=self.intercon.network_diameter(), 
                     prio=True, # ack is prioritized
                     blaze_trail = pkt.get_path()) # do the same about path
        ack.set_sendtime(self.get_now())
        ack.add_to_path(str(self))
        
        if self.node_id == ack.dsthost:
            self.mem_queue.send_pkt(ack)
        else:
            self.interfaces['r'].send_pkt(ack, 0)

        to_rank = pkt.nonreturn_data["to_rank"]
        to_true_rank = pkt.nonreturn_data["to_true_rank"]
        t = pkt.nonreturn_data["type"]
        if to_true_rank not in self.recv_buffer:
            self.recv_buffer[to_true_rank] = []
            from_rank = pkt.nonreturn_data["from_rank"]
            rc = self.comm_world[to_true_rank]['comms'][pkt.nonreturn_data["comm_id"]]
            from_true_rank = get_mpi_true_rank(rc, from_rank)
            self.recv_msgid[to_true_rank] = dict()
            self.recv_msgid[to_true_rank][from_true_rank] = dict()
            self.recv_msgid[to_true_rank][from_true_rank][t] = 0
            msg_id = pkt.nonreturn_data["msg_id"]

            # this message can only be the right one or the one from
            # future; admit anyway
            missing_pieces = set(range(pkt.nonreturn_data["num_pieces"]))
            piece = pkt.nonreturn_data["piece_idx"]
            missing_pieces.remove(piece)
            recvitem = {
                "from_rank" : from_rank,
                "comm_id" : pkt.nonreturn_data["comm_id"],
                "msg_id" : msg_id,
                "type" : t,
                "data_size" : pkt.nonreturn_data["data_size"],
                "missing_pieces" : missing_pieces
            }
            # only piece #0 carries the data
            if piece == 0:
                recvitem["data"] = pkt.nonreturn_data["data"]
            #print("%d: notify_data_recv<1>: put %r in NEW buffer" % (to_true_rank, recvitem))
            self.recv_buffer[to_true_rank].append(recvitem)

        else:
            # linear search is easy
            #print("%d: notify_data_recv<2>: try match: pkt=%s, type=%s" % \
            #      (to_true_rank, pkt, t))
            for idx in range(len(self.recv_buffer[to_true_rank])):
                recvitem = self.recv_buffer[to_true_rank][idx]
                #print("%d: notify_data_recv<2>: recvitem=%r" % (to_true_rank, recvitem))
                # the match criteria:
                # (0) same communicator, and
                # (1) the request has wildcard from_rank or rank is perfect match, and 
                # (2) type is perfect match, or the reqeust has wildcard type and recv item is not reserved, and
                # (3) the recv item does not have msg id or it has the same msg id
                if (recvitem["comm_id"] == pkt.nonreturn_data["comm_id"]) and \
                   (recvitem["from_rank"] is None or recvitem["from_rank"] == pkt.nonreturn_data["from_rank"]) and \
                   (recvitem["type"] == t or recvitem["type"] is None and t[:2] != '__') and \
                   ("msg_id" not in recvitem or recvitem["msg_id"] == pkt.nonreturn_data["msg_id"]):
                    # there can be at most one match
                    break
            else: recvitem = None

            if recvitem is not None:
                if "missing_pieces" in recvitem:
                    # meaning that this is not only a request without
                    # any arrival of data
                    missing_pieces = recvitem["missing_pieces"]
                    piece = pkt.nonreturn_data["piece_idx"]
                    # piece #0 carries the data
                    if piece == 0:
                        recvitem["data"] = pkt.nonreturn_data["data"]
                    # there's possibility the piece has already been
                    # received ealier; that is, this is a duplicate;
                    # otherwise, remove the piece from expected set
                    if piece in missing_pieces:
                        missing_pieces.remove(piece)
                        recvitem["data_size"] += pkt.nonreturn_data["data_size"]
                else:
                    # found a lonely request
                    missing_pieces = set(range(pkt.nonreturn_data["num_pieces"]))
                    piece = pkt.nonreturn_data["piece_idx"]
                    missing_pieces.remove(piece)
                    recvitem["missing_pieces"] = missing_pieces
                    # the first arrival piece must fill in information
                    # except that only piece #0 has the data
                    recvitem["from_rank"] = pkt.nonreturn_data["from_rank"]
                    recvitem["type"] = t
                    recvitem["data_size"] = pkt.nonreturn_data["data_size"]
                    if piece == 0:
                        recvitem["data"] = pkt.nonreturn_data["data"]

                if len(missing_pieces) == 0:
                    # if there's no more missing piece and it's a receive
                    # request, inform the receive process

                    from_rank = recvitem["from_rank"]
                    rc = self.comm_world[to_true_rank]['comms'][recvitem["comm_id"]]
                    from_true_rank = get_mpi_true_rank(rc, from_rank)
                    #print("%d: notify_data_recv<4>: found (full) %r in buffer" % (to_true_rank, recvitem))
                    if "mpi_process" in recvitem:
                        del self.recv_buffer[to_true_rank][idx]
                        if from_true_rank not in self.recv_msgid[to_true_rank]:
                            self.recv_msgid[to_true_rank][from_true_rank] = dict()
                        if t in self.recv_msgid[to_true_rank][from_true_rank]:
                            self.recv_msgid[to_true_rank][from_true_rank][t] += 1
                        else:
                            self.recv_msgid[to_true_rank][from_true_rank][t] = 1
                        #print("%d: recv_msgid[%d][%d][%s]=%d" %
                        #      (to_true_rank, to_true_rank, from_true_rank, t,
                        #       self.recv_msgid[to_true_rank][from_true_rank][t]-1))
                        recvitem["mpi_process"].wake(recvitem)
                    elif "mpi_request" in recvitem:
                        del self.recv_buffer[to_true_rank][idx]
                        if from_true_rank not in self.recv_msgid[to_true_rank]: 
                           self.recv_msgid[to_true_rank][from_true_rank] = dict()
                        if t in self.recv_msgid[to_true_rank][from_true_rank]:
                           self.recv_msgid[to_true_rank][from_true_rank][t] += 1
                        else:
                            self.recv_msgid[to_true_rank][from_true_rank][t] = 1
                        #print("%d: recv_msgid[%d][%d][%s]=%d" %
                        #      (to_true_rank, to_true_rank, from_true_rank, t,
                        #       self.recv_msgid[to_true_rank][from_true_rank][t]-1))
                        recvitem["mpi_request"]["status"] = "success"
                        recvitem["mpi_request"]["from_rank"] = from_rank
                        recvitem["mpi_request"]["type"] = recvitem["type"]
                        recvitem["mpi_request"]["data"] = recvitem["data"]
                        recvitem["mpi_request"]["data_size"] = recvitem["data_size"]
                        if "mpi_process" in recvitem["mpi_request"]:
                            recvitem["mpi_request"]["mpi_process"].wake()

            else:
                # if nothing matches, create one as long as in
                # sequence or from future
                from_rank = pkt.nonreturn_data["from_rank"]
                #print("HERE: rank=%d incoming: comm=%d" % (to_true_rank, pkt.nonreturn_data["comm_id"]))
                rc = self.comm_world[to_true_rank]['comms'][pkt.nonreturn_data["comm_id"]]
                from_true_rank = get_mpi_true_rank(rc, from_rank)
                msg_id = pkt.nonreturn_data["msg_id"]
                if from_true_rank not in self.recv_msgid[to_true_rank]:
                    self.recv_msgid[to_true_rank][from_true_rank] = dict()
                if t not in self.recv_msgid[to_true_rank][from_true_rank]:
                    self.recv_msgid[to_true_rank][from_true_rank][t] = 0
                if self.recv_msgid[to_true_rank][from_true_rank][t] <= msg_id:
                    missing_pieces = set(range(pkt.nonreturn_data["num_pieces"]))
                    piece = pkt.nonreturn_data["piece_idx"]
                    missing_pieces.remove(piece)
                    recvitem = {
                        "from_rank" : from_rank,
                        "comm_id" : pkt.nonreturn_data["comm_id"],
                        "msg_id" : msg_id,
                        "type" : t,
                        "data_size" : pkt.nonreturn_data["data_size"],
                        "missing_pieces" : missing_pieces
                    }
                    if piece == 0:
                        recvitem["data"] = pkt.nonreturn_data["data"]
                    self.recv_buffer[to_true_rank].append(recvitem)
                    #print("%d: notify_data_recv<5>: %r no match in buffer" % (to_true_rank, recvitem))

def send_process(self):
    """A process for (reliably) sending mpi data upon user request."""

    # self is the send proces itself; host is the process' entity
    host = self.entity

    # in the beginning, no send, the process hibernates
    #host.send_semaphore == 0
    self.hibernate()

    while True:
        #print("check: send_process wakes up")
        if len(host.send_buffer) > 0:
            senditem = host.send_buffer.popleft()

            # move it to the resend buffer and schedule resend
            key = host.resend_key
            host.resend_key += 1
            host.resend_buffer[key] = {
                "senditem" : senditem,
                "trials" : 0,
            }
            host.reqService(host.mpi_resend_intv, "resend_mpi_message", key)

            # send it out
            host.send_mpi_message(senditem, key)

        else:
            #print("check: send_process sleeps")
            #host.send_semaphore = 0
            self.hibernate()

def kernel_main_function(self, user_main_function, mpi_comm_world, *arg):
    """Helper function that calls the user's main function."""
    user_main_function(mpi_comm_world, *arg)
