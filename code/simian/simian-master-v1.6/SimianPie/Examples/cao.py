from SimianPie.simian import Simian

simName, startTime, endTime, minDelay, useMPI = "Simulation", 0, 999999999999999999999, 0.0001, True
#simEngine = Simian(simName, startTime, endTime, minDelay, useMPI, "/usr/local/lib/libmpich.so")
simEngine = Simian(simName, startTime, endTime, minDelay, False)

class BBNode(simEngine.Entity):
	def __init__(self, baseInfo, *args):
		super(BBNode, self).__init__(baseInfo)
		self.bbID = args[0]
		self.writeBw = 100
		self.wakeRetval = [-99]
		self.nodeToExecute = dict()
		self.shareCnt = 2 #pre set to 2(2 requests)
		
		#initialize two executor processes to process two requests from two compude nodes
		for i in range(2):
			nodeID = i
			procName = "cpWrite_"+str(nodeID)
                        print("** New process: " + procName)
			reqSize = 100000
			
			self.createProcess(procName, self.cpWrite)
			self.startProcess(procName, procName, 0, nodeID, reqSize)
			#add to node-to-proc dictionary
			procList = None
			if nodeID in self.nodeToExecute:
				procList = self.nodeToExecute[nodeID]
			else:
				procList = []
				self.nodeToExecute[nodeID] = procList

			procList.append(procName)
			

	### Process ###
	def cpWrite(self, this, procName, jobID, nodeID, reqSize):
		bandwidth = self.writeBw / self.shareCnt
		ioTime = reqSize / bandwidth + 2*nodeID + 1
		remainSize = reqSize
		start = self.engine.now
		done = 0
		while done == 0:
			print "bb %d job %d node %d sleep for %f #####" % (self.bbID, jobID, nodeID, ioTime)
			#wait for IO to finish
			this.sleep(ioTime, *(self.wakeRetval))
			print "bb %d job %d node %d wakeup at %f #####" % (self.bbID, jobID, nodeID, simEngine.now)
			if self.wakeRetval[0] == -99:
				#finished naturally
				done = 1
			elif self.wakeRetval[0] == -1:
				#bandwidth change due to change in sharing
				print "bb %d job %d node %d got share cnt change" % (self.bbID, jobID, nodeID)
				retval = self.updateRemainIOTime(bandwidth, remainSize, start)
				ioTime = retval[0]
				remainSize = retval[1]
				start = simEngine.now #reset start for next bw change
				self.wakeRetval[0] = -99 #reset wakeup return value
				print "bb %d job %d node %d iotime after share change %f" % (self.bbID, jobID, nodeID, ioTime)
		procList = self.nodeToExecute[nodeID]
		if procName in procList:
			#remove process from node-to-executor dictionary
			procList.remove(procName)
		self.shareCnt = self.shareCnt - 1
		print "bb %d job %d node %d finished at %f, now change share cnt and notify other executors" % (self.bbID, jobID, nodeID, simEngine.now)
		self.notifyShareCntChange(jobID, nodeID)
		print "bb %d job %d node %d share cnt change notification done" % (self.bbID, jobID, nodeID)

	### Helpers ###
	def notifyShareCntChange(self, jobID, nodeID):
		for i in self.nodeToExecute.keys():
			print "bb %d checking node %d" % (self.bbID, nodeID)
			procList = self.nodeToExecute[i]
			print procList
			if len(procList) > 0:
				for procName in procList:
					procStatus = self.statusProcess(procName)
					if procStatus == "suspended":
						self.wakeRetval[0] = -1
						print "waking up %s" % procName
						self.wakeProcess(procName, *(self.wakeRetval))
		print "bb %d job %d nodeID %d notify share cnt change done" % (self.bbID, jobID, nodeID)

	def updateRemainIOTime(self, oldBw, remainSize, start):
		end = simEngine.now
		remain = abs(remainSize - ((end - start) * oldBw))
		newBw = self.writeBw / self.shareCnt
		newIOTime = remain / newBw
		print "new iotime %f new remain %f" % (newIOTime, remain)
		return [newIOTime, remain]

bbID = 0
args = []
args.append(bbID)
simEngine.addEntity("bb0", BBNode, 0, *args)
simEngine.run()
simEngine.exit()
