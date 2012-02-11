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

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "minilda.h"
#include "random.h"
#include "feature.h"
#include "pcomp.h"
#include "scoped_ptr.h"

namespace minilda {

// This class is an implementation the Gibbs sampling algorithm based on
// Finding scientific topics, Griffiths and Steyvers, PNAS (101), 2004.
class GibbsSamplerImpl : public GibbsSampler {
 public:
  explicit GibbsSamplerImpl() : num_iter_(0), num_docs_(0), num_topics_(0),
                                num_vocabulary_(0), num_word_(0), alpha_(0.0), beta_(0.0),
                                rand_(new Random), features_(new Features) {}
  virtual ~GibbsSamplerImpl() {}

  bool Open(const char* filename);

  bool OpenVocabulary(const char* filename);

  bool SaveWordTopic(const char* filename);

  bool SaveDocumentTopic(const char* filename);

  void Update(int num_iter);

  double** GetPhi();
  double** GetTheta();
  void DestroyPhi(double** phi);
  void DestroyTheta(double** theta);

  void set_num_iter(int n) { num_iter_ = n; }
  void set_num_docs(int n) { num_docs_ = n; }
  void set_num_topics(int n) { num_topics_ = n; }
  void set_num_vocabulary(int n) { num_vocabulary_ = n; }
  void set_num_word(int n) { num_word_ = n; }
  void set_alpha(double alpha) { alpha_ = alpha; }
  void set_beta(double beta) { beta_ = beta; }

  int get_num_iter() const { return num_iter_; }
  int get_num_docs() const { return num_docs_; }
  int get_num_topics() const { return num_topics_; }
  int get_num_vocabulary() const { return num_vocabulary_; }
  int get_num_word() const { return num_word_; }
  double get_alpha() const { return alpha_; }
  double get_beta() const { return beta_; }

  const char* what() { return what_.str().c_str(); }

 private:
  int SelectNextTopic(int doc_id, int word_id);
  void Resample(int token_id);

  void Resize();                        // Resize vectors and matrices
  void Init();                          // Initialize matrices

  int num_iter_;                        // number of iteration on training
  int num_docs_;                        // number of documents (D for short)
  int num_topics_;                      // number of topics (K)
  int num_vocabulary_;                  // number of unique words (W))
  int num_word_;                        // number of word to be saved
  double alpha_;                        // hyper parameter
  double beta_;
  scoped_ptr<Random> rand_;                        // random generator
  scoped_ptr<Features> features_;           // (document_id, word_id) pairs

  std::vector<std::string> words_;
  std::vector<std::vector<int> > word_count_; // W x K matrix
  std::vector<std::vector<int> > doc_count_;  // D x K matrix
  std::vector<int> topic_count_;              // K

  std::vector<double> p_;
  std::vector<int> z_;                  // topic assignments

  std::ostringstream what_;             // logger of this class
};

bool GibbsSamplerImpl::Open(const char* filename) {
  std::ifstream ifs(filename);

  if (!ifs) {
    what_ << "no such file or directory " << filename << std::endl;
    return false;
  }

  int N = 0;
  std::size_t line_num = 0;
  std::string line;

  // Get data size
  std::getline(ifs, line);
  num_docs_ = atoi(line.c_str());

  std::getline(ifs, line);
  num_vocabulary_ = atoi(line.c_str());

  std::getline(ifs, line);
  N = atoi(line.c_str());

  if (N <= 0) {
    what_ << "N is invalid: " << N;
    return false;
  }

  line_num += 3;

  for (int i = 0; i < N; ++i) {
    std::getline(ifs, line);
    if (line[0] == '#' || line.empty()) continue;

    if (features_->Parse(line)) {
      what_ << "line: " << line_num;
    }
    ++line_num;
  }

  Resize();
  Init();

  return true;
}

bool GibbsSamplerImpl::OpenVocabulary(const char* filename) {
  std::ifstream ifs(filename);

  if (!ifs) {
    what_ << "no such file or directory" << filename << std::endl;
    return false;
  }

  std::string line;

  for (int i = 0; i < num_vocabulary_; ++i) {
    std::getline(ifs, line);
    if (line[0] == '#' || line.empty()) continue;
    words_[i] = line;
  }

  return true;
}

bool GibbsSamplerImpl::SaveWordTopic(const char* filename) {
  std::ofstream ofs(filename);
  if (!ofs) {
    what_ << "Cannot open " << filename;
    return false;
  }

  // Set number of word to be saved
  if (num_word_ == 0) num_word_ = num_vocabulary_;

  double** phi = GetPhi();

  for (int k = 0; k < num_topics_; ++k) {
    if (k != 0) ofs << std::endl;
    ofs << "topic: " << k << std::endl;

    std::vector<ProbabilityCompare> ps(num_vocabulary_);

    for (int w = 0; w < num_vocabulary_; ++w) {
      ProbabilityCompare pc;
      pc.id = w;
      pc.value = phi[k][w];
      ps[w] = pc;
    }

    std::sort(ps.begin(), ps.end(), LessProbability());
    for (int i = 0; i < num_word_; ++i) {
      ProbabilityCompare p = ps[i];
      ofs << words_[p.id] << " " << p.value << std::endl;
    }
  }

  ofs.close();
  DestroyPhi(phi);

  return true;
}

bool GibbsSamplerImpl::SaveDocumentTopic(const char* filename) {
  std::ofstream ofs(filename);
  if (!ofs) {
    what_ << "Cannot open " << filename;
    return false;
  }

  double** theta = GetTheta();
  for (int d = 0; d < num_docs_; ++d) {
    for (int k = 0; k < num_topics_; ++k) {
      if (k != 0) ofs << " ";
      ofs << theta[d][k];
    }
    ofs << std::endl;
  }

  ofs.close();
  DestroyTheta(theta);

  return true;
}

void GibbsSamplerImpl::Update(int num_iter) {
  for (int i = 0; i < num_iter; ++i) {
    std::cout << "Number of iteration: " << i << std::endl;
     for (int i = 0; i < features_->size(); ++i) {
       Resample(i);
    }
  }
}

double** GibbsSamplerImpl::GetPhi() {
  double** phi;
  phi = new double*[num_topics_];
  for (int i = 0; i < num_topics_; ++i) phi[i] = new double[num_vocabulary_];

  for (int i = 0; i < num_topics_; ++i) {
    double sum = 0.0;
    for (int j = 0; j < num_vocabulary_; ++j) {
      phi[i][j] = beta_ + word_count_[j][i];
      sum += phi[i][j];
    }
    // Normalize
    double sinv = 1.0 / sum;
    for (int j = 0; j < num_vocabulary_; ++j)
      phi[i][j] *= sinv;
  }
  return phi;
}

double** GibbsSamplerImpl::GetTheta() {
  double** theta;
  theta = new double*[num_docs_];
  for (int i = 0; i < num_docs_; ++i) theta[i] = new double[num_topics_];

  for (int i = 0; i < num_docs_; ++i) {
    double sum = 0.0;
    for (int j = 0; j < num_topics_; ++j) {
      theta[i][j] = alpha_ + doc_count_[i][j];
      sum += theta[i][j];
    }
    // Normalize
    double sinv = 1.0 / sum;
    for (int j = 0; j < num_topics_; ++j) theta[i][j] *= sinv;
  }
  return theta;
}

void GibbsSamplerImpl::DestroyPhi(double** phi) {
  for (int i = 0; i < num_topics_; ++i)
    delete [] phi[i];
  delete [] phi;
}

void GibbsSamplerImpl::DestroyTheta(double** theta) {
  for (int i = 0; i < num_docs_; ++i)
    delete [] theta[i];
  delete [] theta;
}

// Private

int GibbsSamplerImpl::SelectNextTopic(int doc_id, int word_id) {
  for (int k = 0; k < num_topics_; ++k) {
    p_[k] = (word_count_[word_id][k] + beta_)
          * (doc_count_[doc_id][k] + alpha_)
          / (topic_count_[k] + num_vocabulary_ * beta_);
    if (k != 0) p_[k] += p_[k - 1];
  }
  const double u = rand_->gen(1.0) * p_[num_topics_ - 1];
  for (int k = 0; k < num_topics_; ++k) {
    if (u < p_[k]) return k;
  }

  return num_topics_ - 1;
}

void GibbsSamplerImpl::Resample(int token_id) {
  const int d = features_->doc_id(token_id);
  const int w = features_->word_id(token_id);
  int assign = z_[token_id];

  --word_count_[w][assign];
  --doc_count_[d][assign];
  --topic_count_[assign];

  assign = SelectNextTopic(d, w);
  ++word_count_[w][assign];
  ++doc_count_[d][assign];
  ++topic_count_[assign];
  z_[token_id] = assign;
}

void GibbsSamplerImpl::Resize() {
  words_.resize(num_vocabulary_);

  word_count_.resize(num_vocabulary_);
  for (int i = 0; i < num_vocabulary_; ++i) {
    word_count_[i].resize(num_topics_);
    std::fill(word_count_[i].begin(), word_count_[i].end(), 0);
  }

  doc_count_.resize(num_docs_);
  for (int i = 0; i < num_docs_; ++i) {
    doc_count_[i].resize(num_topics_);
    std::fill(doc_count_[i].begin(), doc_count_[i].end(), 0);
  }

  topic_count_.resize(num_topics_);
  std::fill(topic_count_.begin(), topic_count_.end(), 0);

  p_.resize(num_topics_);
  std::fill(p_.begin(), p_.end(), 0.0);

  z_.resize(features_->size());
  std::fill(z_.begin(), z_.end(), 0);
}

void GibbsSamplerImpl::Init() {
  for (int i = 0; i < features_->size(); ++i) {
    int d = features_->doc_id(i);
    int w = features_->word_id(i);
    int assign = rand_->gen(num_topics_);

    ++word_count_[w][assign];
    ++doc_count_[d][assign];
    ++topic_count_[assign];
    z_[i] = assign;
  }
}

GibbsSampler* GibbsSampler::instance() {
  return new GibbsSamplerImpl;
}

} // namespace minilda
