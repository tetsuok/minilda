// Copyright (c) 2010, 2012 Tetsuo Kiso
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above
//    copyright notice, this list of conditions and the
//    following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the
//    following disclaimer in the documentation and/or other
//    materials provided with the distribution.
//
//  * Neither the name of Tetsuo Kiso nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef MINILDA_FEATURE_H_
#define MINILDA_FEATURE_H_

#include <sstream>
#include <cassert>
#include <vector>

namespace minilda {

struct Token {
  int doc_id;
  int word_id;
};

class Features {
 public:
  Features() {}
  ~Features() {}

  bool Parse(const std::string& line) {
    std::istringstream is(line);
    int doc_id = 0;
    int word_id = 0;
    int count = 0;
    is >> doc_id >> word_id >> count;

    if (!doc_id || !word_id || !count) {
      return false;
    }

    {
      for (int i = 0; i < count; ++i) {
        Token t;
        t.doc_id = doc_id - 1;
        t.word_id = word_id - 1;
        features_.push_back(t);
      }
    }
    return true;
  }

  const Token* get_token(int i) {
    assert(i < features_.size());
    return &features_[i];
  }

  int doc_id(int i) { return features_[i].doc_id; }
  int word_id(int i) { return features_[i].word_id; }

  int size() const { return features_.size(); }
  void resize(int size) { features_.resize(size); }

 private:
  std::vector<Token> features_;
};
} // namespace minilda

#endif  // MINILDA_FEATURE_H_
