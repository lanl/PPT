/*
Author: Nandakishore Santhi
Date: Nov 2016
*/
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Support/raw_ostream.h>

using namespace std;
using namespace llvm;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << "bitcode.[llvm|bc]" << endl;
        return 1;
    }


    vector<string> funNameList; //To hold user defined function names in the program
    unordered_map<string, int> funCallCount;  //To hold the count of number of times each user defined function is called
    int i = 0;

    LLVMContext &Context = getGlobalContext();
    SMDiagnostic Err;
    Module *Mod = ParseIRFile(argv[1], Err, Context);

    if (!Mod) {
        Err.print(argv[0], errs());
        return 1;
    }

    cout << "Module : " << Mod->getModuleIdentifier() << endl;
    cout << "\tTargetTriple :" << endl;
    cout << "\t\t" << Mod->getTargetTriple() << endl;

    for (auto iter1 = Mod->getFunctionList().begin(); iter1 != Mod->getFunctionList().end(); iter1++) {
        string calledFunName;
        Function &f = *iter1;
        if((!f.isDeclaration()) && (!f.isIntrinsic())){
            calledFunName = f.getName().str();
            funNameList.push_back(calledFunName);
        }

        funCallCount.insert(make_pair(calledFunName, 0));
    }

    for (auto iter1 = Mod->getFunctionList().begin(); iter1 != Mod->getFunctionList().end(); iter1++) {
        Function &f = *iter1;

        cout << "\n\tFunction <@ " << &f << "> : " << f.getName().str() << " : " << (f.isDeclaration() ? "Declaration" : "Definition") << " : " << (f.isIntrinsic() ? "Intrinsic" : "NonIntrinsic") << endl;
        cout << "\t\tArgList :" << endl;
        for (auto iterA = f.getArgumentList().begin(); iterA != f.getArgumentList().end(); iterA++) {
            Value &arg = *iterA;
            string argName;
            if (arg.hasName()) argName = arg.getName().str();
            else { // If no explicit name for Argument, then get an implicit name instead
                raw_string_ostream rso(argName);
                arg.printAsOperand(rso, true);
            }
            cout << "\t\t\t" << argName << "<@ " << &arg << ">" << endl;
        }


        for (auto iter2 = f.getBasicBlockList().begin(); iter2 != f.getBasicBlockList().end(); iter2++) {
            BasicBlock &bb = *iter2;
            string bbName;
            int subBlockNum=0;

            if (bb.hasName()) bbName = bb.getName().str();
            else { // If no explicit name for Basicblock, then get an implicit name instead
                raw_string_ostream rso(bbName);
                bb.printAsOperand(rso, true);
            }
            cout << "\n\t\tBasicBlock <@ " << &bb << "> : " << bbName << endl;
            subBlockNum++;

            for (auto iter3 = bb.begin(); iter3 != bb.end(); iter3++) {
                Instruction &inst = *iter3;
                const DebugLoc &location = inst.getDebugLoc();
                string opCodeNameStr = inst.getOpcodeName();

                if((opCodeNameStr.compare("call") == 0)) {
                    int flag = 0;
                    string calledFunName;

                    for(unsigned int j=0; j<inst.getNumOperands(); j++) {
                        Value &opnd = *inst.getOperand(j);
                        string opndName;

                        if (opnd.hasName()) opndName = opnd.getName().str();

                        for (unsigned k=0; k<funNameList.size(); k++){
                            if(opndName == funNameList[k]){
                                calledFunName = opndName;
                                flag = 1;
                                break;
                            }
                        }
                    }

                    if (flag) {
                        funCallCount[calledFunName]++;

                        cout << "\t\t\tInstruction <@ " << &inst << "> : (";
                        if (!location.isUnknown()) cout << location.getLine() << ", " << location.getCol();
                        cout << ") : br\n\t\t\t\tlabel : " << bbName << "." << subBlockNum << "<@" << &bb << "." << subBlockNum << ">" << endl;

                        cout << "\n\t\tBasicBlock <@ " << &bb << "." << subBlockNum << "> : " << bbName << "." << subBlockNum << endl;
                        subBlockNum++;
                    }
                }

                cout << "\t\t\tInstruction <@ " << &inst << "> : (";
                if (!location.isUnknown()) cout << location.getLine() << ", " << location.getCol();
                else cout << "0, 0"; // ie., not from the source
                cout << ") : " << inst.getOpcodeName() << endl;

                for(unsigned int i=0; i<inst.getNumOperands(); i++) {
                    Value &opnd = *inst.getOperand(i);
                    string opndName;
                    if (opnd.hasName()) opndName = opnd.getName().str();
                    else { // If no explicit name for Operand, then get an implicit name instead
                        raw_string_ostream rso(opndName);
                        opnd.printAsOperand(rso, true);
                    }
                    string typeName;
                    raw_string_ostream rsoT(typeName);
                    opnd.getType()->print(rsoT);
                    cout << "\t\t\t\t" << rsoT.str() << " : " << opndName << "<@ " << &opnd << ">" << endl;
                }

                if((opCodeNameStr.compare("call") == 0)) {
                    int flag = 0;
                    string calledFunName;

                    for(unsigned int j=0; j<inst.getNumOperands(); j++) {
                        Value &opnd = *inst.getOperand(j);
                        string opndName;

                        if (opnd.hasName()) opndName = opnd.getName().str();

                        for (unsigned k=0; k<funNameList.size(); k++){
                            if(opndName == funNameList[k]){
                                calledFunName = opndName;
                                flag = 1;
                                break;
                            }
                        }
                    }

                    if (flag) {
                        funCallCount[calledFunName]++;

                        cout << "\t\t\tInstruction <@ " << &inst << "> : (";
                        if (!location.isUnknown()) cout << location.getLine() << ", " << location.getCol();
                        cout << ") : br\n\t\t\t\tlabel : " << bbName << "." << subBlockNum << "<@" << &bb << "." << subBlockNum << ">" << endl;

                        cout << "\n\t\tBasicBlock <@ " << &bb << "." << subBlockNum << "> : " << bbName << "." << subBlockNum << endl;
                        subBlockNum++;
                    }
                }
            }
        }
    }
    return 0;
}
