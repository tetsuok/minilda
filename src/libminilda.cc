// Copyright (c) 2010, Tetsuo Kiso
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

#include <iostream>
#include <cstdlib>
#include "minilda.h"
#include "scoped_ptr.h"
#include "timer.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

struct minilda_gibbs_sampler_t {
  minilda::GibbsSampler *ptr;
};

// C APIs

minilda_gibbs_sampler_t* minilda_gibbs_sampler_new() {
  minilda_gibbs_sampler_t *learner = new minilda_gibbs_sampler_t;
  learner->ptr = minilda::GibbsSampler::instance();
  return learner;
}

void minilda_gibbs_sampler_destroy(minilda_gibbs_sampler_t *learner) {
  delete learner->ptr;
  delete learner;
  learner = 0;
}

int minilda_gibbs_sampler_open(minilda_gibbs_sampler_t *learner,
                               const char *filename) {
  return learner->ptr->Open(filename);
}

int minilda_gibbs_sampler_open_vocabulary(minilda_gibbs_sampler_t *learner,
                                          const char *filename) {
  return learner->ptr->OpenVocabulary(filename);
}

int minilda_gibbs_sampler_save_word_topic(minilda_gibbs_sampler_t *learner,
                                          const char *filename) {
  return learner->ptr->SaveWordTopic(filename);
}

int minilda_gibbs_sampler_save_document_topic(minilda_gibbs_sampler_t *learner,
                                              const char *filename) {
  return learner->ptr->SaveDocumentTopic(filename);
}

void minilda_gibbs_sampler_update(minilda_gibbs_sampler_t *learner,
                                  int num_iter) {
  learner->ptr->Update(num_iter);
}

double** minilda_gibbs_sampler_get_phi(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->GetPhi();
}

double** minilda_gibbs_sampler_get_theta(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->GetTheta();
}

void minilda_gibbs_sampler_destroy_phi(minilda_gibbs_sampler_t *learner,
                                       double **phi) {
  learner->ptr->DestroyPhi(phi);
}

void minilda_gibbs_sampler_destroy_theta(minilda_gibbs_sampler_t *learner,
                                         double **theta) {
  learner->ptr->DestroyTheta(theta);
}

void minilda_gibbs_sampler_set_num_iter(minilda_gibbs_sampler_t *learner,
                                        int num_iter) {
  learner->ptr->set_num_iter(num_iter);
}

void minilda_gibbs_sampler_set_num_docs(minilda_gibbs_sampler_t *learner,
                                        int num_docs) {
  learner->ptr->set_num_docs(num_docs);
}

void minilda_gibbs_sampler_set_num_topics(minilda_gibbs_sampler_t *learner,
                                          int num_topics) {
  learner->ptr->set_num_topics(num_topics);
}

void minilda_gibbs_sampler_set_num_vocabulary(minilda_gibbs_sampler_t *learner,
                                              int num_vocabulary) {
  learner->ptr->set_num_vocabulary(num_vocabulary);
}

void minilda_gibbs_sampler_set_num_word(minilda_gibbs_sampler_t *learner,
                                        int num_word) {
  learner->ptr->set_num_word(num_word);
}

void minilda_gibbs_sampler_set_alpha(minilda_gibbs_sampler_t *learner,
                                     double alpha) {
  learner->ptr->set_alpha(alpha);
}

void minilda_gibbs_sampler_set_beta(minilda_gibbs_sampler_t *learner,
                                    double beta) {
  learner->ptr->set_beta(beta);
}

int minilda_gibbs_sampler_get_num_iter(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->get_num_iter();
}

int minilda_gibbs_sampler_get_num_docs(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->get_num_docs();
}

int minilda_gibbs_sampler_get_num_topics(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->get_num_topics();
}

int minilda_gibbs_sampler_get_num_vocabulary(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->get_num_vocabulary();
}

int minilda_gibbs_sampler_get_num_word(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->get_num_word();
}

double minilda_gibbs_sampler_get_alpha(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->get_alpha();
}

double minilda_gibbs_sampler_get_beta(minilda_gibbs_sampler_t *learner) {
  return learner->ptr->get_beta();
}

const char* minilda_gibbs_sampler_error(minilda_gibbs_sampler_t* learner) {
  return learner->ptr->what();
}

namespace {

void Usage() {
  std::cerr << "Usage: lda_learn"
            << " vocab_file model -K=num_topics -I=num_iter -W=num_word\n"
            << "Option:\n"
            << "-K: number of topic (Default 50)\n"
            << "-I: number of iteration (Default 200)\n"
            << "-W: number of words in vocabrary to be output (Default all)\n";
}

inline int ParseOption(const char* s, int& K, int& num_iter, int& num_word) {
  std::string str(s);

  if (str.size() < 2) return 0;
  if (str.substr(0, 3) == "-K=") {
    K = atoi(str.substr(3).c_str());
    return 0;
  }
  else if (str.substr(0, 3) == "-I=") {
    num_iter = atoi(str.substr(3).c_str());
    return 0;
  }
  else if (str.substr(0, 3) == "-W=") {
    num_word = atoi(str.substr(3).c_str());
    return 0;
  }
  else if (str.substr(0, 2) == "-h") {
    Usage();
    exit(1);
  }
  else {
    std::cerr << "Unknown option: " << str.substr(0,3) << std::endl;
    return -1;
  }
}

} // namespace

int lda_learn(int argc, char** argv) {
  using namespace minilda;
  if (argc < 4) {
    Usage();
    return -1;
  }

  int num_iter = 200;                   // number of iteration
  int K = 50;                           // number of latent topics
  int num_word = 0;

  for (int i = 1; i < 4; ++i) {
    if (*argv[i] == '-') {
      Usage();
      return -1;
    }
  }

  for (int i = 4; i < argc; ++i) {
    if (ParseOption(argv[i], K, num_iter, num_word) != 0) {
      std::cerr << "Cannot parse argv" << std::endl;
      return -1;
    }
  }

  std::string train_file = argv[1];
  std::string vocab_file = argv[2];
  std::string model_file = argv[3];

  std::string model_pwz = model_file + ".pwz";
  std::string model_pzd = model_file + ".pzd";

  TinyTimer timer;

  scoped_ptr<GibbsSampler> gibbs_sampler(GibbsSampler::instance());
  gibbs_sampler->set_num_topics(K);
  gibbs_sampler->set_alpha(50.0 / K);
  gibbs_sampler->set_beta(0.1);
  gibbs_sampler->set_num_word(num_word);

  if (!gibbs_sampler->Open(train_file.c_str())) {
    std::cerr << "cannot open " << train_file
              << gibbs_sampler->what() << std::endl;
    return -1;
  }

  if (!gibbs_sampler->OpenVocabulary(vocab_file.c_str())) {
    std::cerr << "cannot open " << vocab_file
              << gibbs_sampler->what() << std::endl;
    return -1;
  }

  gibbs_sampler->Update(num_iter);

  if (!gibbs_sampler->SaveWordTopic(model_pwz.c_str())) {
    std::cerr << "Cannot save " << model_pwz
              << gibbs_sampler->what() << std::endl;
    return -1;
  }

  if (!gibbs_sampler->SaveDocumentTopic(model_pzd.c_str())) {
    std::cerr << "Cannot save " << model_pzd
              << gibbs_sampler->what() << std::endl;
    return -1;
  }

  std::printf("Done!\nTime: %.4f sec.\n", timer.GetElapsedTime());

  return 0;
}

int lda_infer(int argc, char** argv) {
  // Write inference programs here.
  return 0;
}
