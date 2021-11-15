#!/usr/bin/env python3
import sys
import os
import yaml

registerUsage = {}
functionName = None

os.system("ptxas -v " + sys.argv[-2] + " 2> tmp_ptx.out")

with open("tmp_ptx.out", "r") as ptxfile:
    for line in ptxfile:
        if functionName is None:
            if line.find("Function properties for") > 0:
                functionName = line.split(" ")[-1][:-1]
        else:
            if line.find("registers,") > 0:
                registerUsage[functionName] = line.split(" ")[7]
                functionName = None
os.remove('tmp_ptx.out')
os.remove('elf.o')
print(registerUsage)
doc = yaml.load(open(sys.argv[-1], "r"))
for function in doc:
    if function['Name'] in registerUsage:
        function['RegisterUsage'] = int(registerUsage[function['Name']])
yaml.dump(doc, open(sys.argv[-1], 'w'), default_flow_style=False)
