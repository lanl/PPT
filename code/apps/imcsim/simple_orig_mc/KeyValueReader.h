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

#ifndef __KEY_VALUE_READER_H
#define __KEY_VALUE_READER_H

#include <string>

namespace CKG_Utils {

class KeyValueReader final
{
public:
    
    // Status error values
    enum Exception {
        ExceptionAlreadyReadAFile,
        ExceptionFileNotRead,
        ExceptionOpenFileError,
        ExceptionParseFileError,
        ExceptionKeyNotFound,
        ExceptionStringConversionError
    };
    
    // Constructor/Destructor
    explicit KeyValueReader();
    ~KeyValueReader();
    
    // Don't allow copy or assignment operators
    KeyValueReader(const KeyValueReader &kvr) = delete;
    KeyValueReader &operator=(const KeyValueReader &kvr) = delete;
    
    // Interface
    void readFile(const std::string &filename);
    void getString(const std::string &key, std::string &value) const;
    void getInt(const std::string &key, int &value) const;
    void getDouble(const std::string &key, double &value) const;
    void getFloat(const std::string &key, float &value) const;
    void getBool(const std::string &key, bool &value) const;
    void print() const;
    void reset();
    
private:
    struct Private;
    Private *c_data;
};

} // End namespace

#endif
