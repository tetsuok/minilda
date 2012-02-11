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

#ifndef MINILDA_MINILDA_H_
#define MINILDA_MINILDA_H_

// C interface
#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MINILDA_DLL_EXTERN
#define MINILDA_DLL_EXTERN extern
#endif

// C APIs
#ifndef SWIG
  typedef struct minilda_gibbs_sampler_t minilda_gibbs_sampler_t;

  // C interface
  MINILDA_DLL_EXTERN int lda_learn(int argc, char** argv);

  MINILDA_DLL_EXTERN int lda_infer(int argc, char** argv);

  // Gibbs sampler
  MINILDA_DLL_EXTERN minilda_gibbs_sampler_t* minilda_gibbs_sampler_new();

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_destroy(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_open(minilda_gibbs_sampler_t *learner,
                                                    const char *filename);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_open_vocabulary(minilda_gibbs_sampler_t *learner,
                                                               const char *filename);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_save_word_topic(minilda_gibbs_sampler_t *learner,
                                                               const char *filename);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_save_document_topic(minilda_gibbs_sampler_t *learner,
                                                                   const char *filename);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_update(minilda_gibbs_sampler_t *learner,
                                                       int num_iter);

  MINILDA_DLL_EXTERN double** minilda_gibbs_sampler_get_phi(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN double** minilda_gibbs_sampler_get_theta(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_destroy_phi(minilda_gibbs_sampler_t *learner,
                                                            double **phi);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_destroy_theta(minilda_gibbs_sampler_t *learner,
                                                              double **theta);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_set_num_iter(minilda_gibbs_sampler_t *learner,
                                                             int num_iter);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_set_num_docs(minilda_gibbs_sampler_t *learner,
                                                             int num_docs);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_set_num_topics(minilda_gibbs_sampler_t *learner,
                                                               int num_topics);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_set_num_vocabulary(minilda_gibbs_sampler_t *learner,
                                                                   int num_vocabulary);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_set_num_word(minilda_gibbs_sampler_t *learner,
                                                             int num_word);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_set_alpha(minilda_gibbs_sampler_t *learner,
                                                          double alpha);

  MINILDA_DLL_EXTERN void minilda_gibbs_sampler_set_beta(minilda_gibbs_sampler_t *learner,
                                                         double beta);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_get_num_iter(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_get_num_docs(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_get_num_topics(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_get_num_vocabulary(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN int minilda_gibbs_sampler_get_num_word(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN double minilda_gibbs_sampler_get_alpha(minilda_gibbs_sampler_t *learner);

  MINILDA_DLL_EXTERN double minilda_gibbs_sampler_get_beta(minilda_gibbs_sampler_t *learner);
#endif  // SWIG

#ifdef __cplusplus
}
#endif

// C++ interface
#ifdef __cplusplus

namespace minilda {

class GibbsSampler {
 public:
  virtual ~GibbsSampler() {}

  // Open bow file.
  virtual bool Open(const char* filename) = 0;

  // Open vocabulary data.
  virtual bool OpenVocabulary(const char* filename) = 0;

  // Save word probability by topic
  virtual bool SaveWordTopic(const char* filename) = 0;

  // Save document probability by topic
  virtual bool SaveDocumentTopic(const char* filename) = 0;

  // Update parameters over the iterations
  virtual void Update(int num_iter) = 0;

  // Get phi.
  virtual double** GetPhi() = 0;

  // Get theta.
  virtual double** GetTheta() = 0;

  // Destroy phi
  virtual void DestroyPhi(double** phi) = 0;

  // Destroy theta
  virtual void DestroyTheta(double** theta) = 0;

  // Set number of iteration.
  virtual void set_num_iter(int n) = 0;

  virtual void set_num_docs(int n) = 0;

  // Set number of topics.
  virtual void set_num_topics(int n) = 0;

  // Set number of vocabularies
  virtual void set_num_vocabulary(int n) = 0;

  // Set number of words to be saved
  virtual void set_num_word(int n) = 0;

  // Set hyper parameter alpha
  virtual void set_alpha(double alpha) = 0;

  // Set hyper parameter beta
  virtual void set_beta(double beta) = 0;

  // Get number of iteration.
  virtual int get_num_iter() const = 0;

  // Get number of documents
  virtual int get_num_docs() const = 0;

  // Get number of topics.
  virtual int get_num_topics() const = 0;

  // Get number of unique words.
  virtual int get_num_vocabulary() const = 0;

  // Get number of words to be saved
  virtual int get_num_word() const = 0;

  // Get hyper parameter alpha
  virtual double get_alpha() const = 0;

  // Get hyper parameter beta
  virtual double get_beta() const = 0;

  // Get instance
  static GibbsSampler *instance();
};
} // namespace minilda

#endif  // __cplusplus
#endif  // MINILDA_MINILDA_H_
