/*
The MIT License (MIT)

Copyright (c) 2016 C. Kristopher Garrett

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "KeyValueReader.h"
#include <vector>
#include <fstream>
#include <stdio.h>

using namespace std;

static const int KEY_NOT_FOUND = -1;

namespace CKG_Utils {


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//                              Helper Functions
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*
    getRidOfComments
    
    Deletes # and everything after that in the string.
*/
static
void getRidOfComments(string &line)
{
    size_t commentStart = line.find('#');
    if (commentStart != string::npos)
        line = line.substr(0, commentStart);
}


/*
    deleteWhitespace
    
    Deletes all the initial whitespace in the string.
    Returns number of whitespace characters.
*/
static
size_t deleteWhitespace(string &line)
{
    static const string whitespace = " \t";
    size_t pos = 0;
    
    for (pos = 0; pos < line.size(); pos++) {
        if (whitespace.find(line[pos]) == string::npos) 
            break;
    }
    line = line.substr(pos);
    return pos;
}


/*
    areStringsEqual
    
    Returns true if the two strings are equal.
    Neglects upper/lower case.
*/
static
bool areStringsEqual(const string &s1, const string &s2)
{
    if (s1.size() != s2.size())
        return false;
    
    for (size_t i = 0; i < s1.size(); i++) {
        if (toupper(s1[i]) != toupper(s2[i]))
            return false;
    }
    
    return true;
}


/*
    popToken
    
    A token is made up of a-z, A-Z, 0-9, and {+,-,_,.}.
    Takes the token out of 'line' and puts it in 'token'.
*/
static
void popToken(string &line, string &token)
{
    static const string allowableChar = "abcdefghijklmnopqrstuvwxyz"
                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "0123456789"
                                        "+-_.";
    size_t pos = 0;
    token = "";
    
    for (pos = 0; pos < line.size(); pos++) {
        if (allowableChar.find(line[pos]) == string::npos) 
            break;
    }
    token = line.substr(0, pos);
    line = line.substr(pos);
}


/*
    parseLine
    
    From a line of input, sets outKey and outValue.
    If the line parses correctly, function returns true.
    Otherwise function returns false.
    
    In the case of true return and the line contains a key/value pair, outKey and
    outValue contain that data.
    In the case of true return and the line contains only whitespace or comments,
    outKey and outValue are set to "".
    In the case of false return, outKey and outValue are set to "".
*/
static
bool parseLine(const string &line, string &outKey, string &outValue)
{
    string parsedLine = line;
    size_t numWhitespace = 0;
    string key = "";
    string value = "";
    
    // Set outKey, outValue to default ""
    outKey = "";
    outValue = "";
    
    // Get rid of comments in line
    getRidOfComments(parsedLine);
    
    // Delete whitespace
    deleteWhitespace(parsedLine);
    
    // Check for empty line
    if (parsedLine.size() == 0) {
        return true;
    }
    
    // Get key
    popToken(parsedLine, key);
    if (key == "") {
    	return false;
    }
    
    // Delete whitespace
    numWhitespace = deleteWhitespace(parsedLine);
    if (numWhitespace == 0) {
    	return false;
    }
    
    // Get value
    popToken(parsedLine, value);
    if (value == "") {
    	return false;
    }
    
    // Delete whitespace
    deleteWhitespace(parsedLine);
    
    // Check for non-empty line
    if (parsedLine.size() != 0) {
    	return false;
    }
    
    // If here, the line parsed correctly
    outKey = key;
    outValue = value;
    return true;
}


/*
    findKey
    
    Returns the index of the key in the keyVector.
    If key is not found, returns KEY_NOT_FOUND.
*/
static
int findKey(const vector<string> &keyVector, const string &key)
{
    for (size_t i = 0; i < keyVector.size(); i++) {
        if (areStringsEqual(key, keyVector[i])) {
            return i;
        }
    }
    
    return KEY_NOT_FOUND;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//                              Private Class Data
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*
    PrivateData class
*/
struct KeyValueReader::Private
{
    void printMessage(const string &message);
    
    vector<string> c_keyVector;
    vector<string> c_valueVector;
    bool c_isFileRead;
    string c_filename;
};


/*
    printMessage
    
    Causes an abort of the program due to an error.
*/
void KeyValueReader::Private::printMessage(const string &message)
{
    printf("KeyValueReader error in file %s\n", c_filename.c_str());
    printf("   %s\n", message.c_str());
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//                              Class Implementation
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*
    Constructor/Destructor
*/
KeyValueReader::KeyValueReader()
{
    c_data = new Private();
    reset();
}
KeyValueReader::~KeyValueReader()
{
    delete c_data;
}


/*
    reset
    
    Resets the KeyValueReader to an uninitialized state.
*/
void KeyValueReader::reset()
{
	c_data->c_isFileRead = false;
    c_data->c_filename = "";
    c_data->c_keyVector.clear();
    c_data->c_valueVector.clear();
}


/*
    readFile
    
    Reads in a key-value file.
    Throws an error if appropriate.
*/
void KeyValueReader::readFile(const string &filename)
{
    ifstream file;
    string line = "";
    int lineNum = 1;
    bool totalParseOk = true;
    string oldFilename = c_data->c_filename;
    
    
    // Set filename (used in printMessage)
    c_data->c_filename = filename;
    
    
    // Check if already read a file
    if (c_data->c_isFileRead) {
        c_data->printMessage("Already read a file");
        c_data->c_filename = oldFilename;
        throw ExceptionAlreadyReadAFile;
    }
    
    
    // Open file
    file.open(filename);
    if (!file.is_open()) {
        c_data->printMessage("Could not open file");
        c_data->c_filename = oldFilename;
        throw ExceptionOpenFileError;
    }
    
    
    // Parse each line
    while (std::getline(file, line)) {
        string key = "";
        string value = "";
        bool parseOk = parseLine(line, key, value);
        
        // Check parse is ok
        if (!parseOk) {
            totalParseOk = false;
            string errorString = "Parse error on line ";
            errorString += to_string(lineNum);
            c_data->printMessage(errorString);
        }
        
        // Check if duplicate key
        else if (findKey(c_data->c_keyVector, key) != KEY_NOT_FOUND) {
            totalParseOk = false;
        	string errorString = "Duplicate key on line ";
            errorString += to_string(lineNum);
            c_data->printMessage(errorString);
        }
        
        // Add key/value if line had one
        else if (key != "") {
            c_data->c_keyVector.push_back(key);
            c_data->c_valueVector.push_back(value);
        }
        lineNum++;
    }
    
    
    // Close file and return
    file.close();
    c_data->c_isFileRead = true;
    
    if (!totalParseOk)
        throw ExceptionParseFileError;
}


/*
    getString
    
    Gets string value from key.
    Throws an error if appropriate.
    If an error occurs, value is set to "".
*/
void KeyValueReader::getString(const std::string &key, std::string &value) const
{
	int keyIndex = 0;
	
	// Default value
    value = "";
	
	// Check for file read
    if (!c_data->c_isFileRead) {
    	c_data->printMessage("File not read.");
    	throw ExceptionFileNotRead;
    }
    
    // Find value from key
    keyIndex = findKey(c_data->c_keyVector, key);
    if (keyIndex == KEY_NOT_FOUND) {
    	c_data->printMessage("Key not found");
    	throw ExceptionKeyNotFound;
    }
    
    value = c_data->c_valueVector[keyIndex];
}


/*
    getInt
    
    Gets integer value from key.
    Throws an error if appropriate.
    If an error occurs, value is set to zero.
*/
void KeyValueReader::getInt(const std::string &key, int &value) const
{
    string valueString;
    
    // Default value
    value = 0;
    
    // Get value as string
    getString(key, valueString);
    
    // Convert to int
    try {
        value = stoi(valueString);
    }
    catch (...) {
        c_data->printMessage("Error converting value to int");
        throw ExceptionStringConversionError;
    }
}


/*
    getDouble
    
    Gets double value from key.
    Throws an error if appropriate.
    If an error occurs, value is set to zero.
*/
void KeyValueReader::getDouble(const std::string &key, double &value) const
{
    string valueString;
    
    // Default value
    value = 0.0;
    
    // Get value as string
    getString(key, valueString);
    
    // Convert to double
    try {
        value = stod(valueString);
    }
    catch (...) {
        c_data->printMessage("Error converting value to double");
        throw ExceptionStringConversionError;
    }
}


/*
    getFloat
    
    Gets float value from key.
    Throws an error if appropriate.
    If an error occurs, value is set to zero.
*/
void KeyValueReader::getFloat(const std::string &key, float &value) const
{
    string valueString;
    
    // Default value
    value = 0.0f;
    
    // Get value as string
    getString(key, valueString);
    
    // Convert value to float
    try {
        value = stof(valueString);
    }
    catch (...) {
        c_data->printMessage("Error converting value to float");
        throw ExceptionStringConversionError;
    }
}


/*
    getBool
    
    Gets boolean value from key.
    Throws an error if appropriate.
    If an error occurs, value is set to false.
*/
void KeyValueReader::getBool(const std::string &key, bool &value) const
{
    string valueString;
    
    // Default value
    value = false;
    
    // Get value from key as string
    getString(key, valueString);
    
    // Check for true
    if (areStringsEqual(valueString, "true")) {
        value = true;
        return;
    }
    
    // Check for false
    if (areStringsEqual(valueString, "false")) {
        value = false;
        return;
    }
    
    // String conversion error if we get here
    c_data->printMessage("Error converting value to bool");
    throw ExceptionStringConversionError;
}


/*
    print
    
    Prints entire set of key/value pairs.
    Throws an error if appropriate.
*/
void KeyValueReader::print() const
{
    printf("\n--- KeyValueReader Data (%s) ---\n", c_data->c_filename.c_str());
    
    // Check for file read
    if (!c_data->c_isFileRead) {
    	c_data->printMessage("File not read.");
    	throw ExceptionFileNotRead;
    }
    
    // Print KeyValueReader data
    for (size_t i = 0; i < c_data->c_keyVector.size(); i++) {
        printf("    %s %s\n", c_data->c_keyVector[i].c_str(), 
                              c_data->c_valueVector[i].c_str());
    }
    printf("\n");
}


} // End namespace

