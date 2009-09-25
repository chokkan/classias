/*
 *		Classias library.
 *
 * Copyright (c) 2008,2009 Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id$ */

#ifndef __CLASSIAS_CLASSIAS_H__
#define __CLASSIAS_CLASSIAS_H__

#include <vector>
#include "types.h"
#include "feature_generator.h"
#include "instance.h"
#include "data.h"

namespace classias
{

typedef std::vector<double> weight_vector;
typedef default_vector<double> expandable_weight_vector;

typedef dense_feature_generator_base<int, int, int> dense_feature_generator;
typedef sparse_feature_generator_base<int, int, int> sparse_feature_generator;
typedef thru_feature_generator_base<int, int, int> thru_feature_generator;

typedef sparse_vector_base<int, double> sparse_attributes;

typedef binary_instance_base<sparse_attributes> binstance;
typedef binary_data_base<binstance> bdata;
typedef binary_data_with_quark_base<binstance, quark> bsdata;

typedef candidate_instance_base<sparse_attributes> cinstance;
typedef candidate_data_base<cinstance, thru_feature_generator> cdata;
typedef candidate_data_with_quark_base<cinstance, quark, quark, thru_feature_generator> csdata;

typedef multi_instance_base<sparse_attributes> minstance;
typedef multi_data_base<minstance, dense_feature_generator> mdata;
typedef multi_data_with_quark_base<minstance, quark, quark, dense_feature_generator> msdata;

typedef multi_instance_base<sparse_attributes> ninstance;
typedef multi_data_base<ninstance, sparse_feature_generator> ndata;
typedef multi_data_with_quark_base<ninstance, quark, quark, sparse_feature_generator> nsdata;

};

/**
@mainpage Classias: C++ Template Class Library for Classification

@section sample Sample programs
- @ref sample_train_binary_batch "Training a binary classifier on a data set"
- @ref sample_train_binary_online "Training a binary classifier with real online setting"
- @ref sample_tag_binary "Binary classifier"

@section class_ref Class reference
- Instance types
    - Binary instance:
        \ref classias::binary_instance_base
    - Multi-class instance:
        \ref classias::multi_instance_base
    - Candidate instance:
        \ref classias::candidate_instance_base
- Data set
    - Binary data set:
        \ref classias::binary_data_base
    - Binary data set with a string quark for attributes:
        \ref classias::binary_data_with_quark_base
    - Multi-class data set:
        \ref classias::multi_data_base
    - Multi-class data set with string quarks for attributes and labels:
        \ref classias::multi_data_with_quark_base
    - Candidate data set:
        \ref classias::candidate_data_base
    - Candidate data set with string quarks for attributes and labels:
        \ref classias::candidate_data_with_quark_base
- Feature generators
    - Dummy feature generator for candidate instances:
        \ref classias::thru_feature_generator_base
    - Dense feature generator:
        \ref classias::dense_feature_generator_base
    - Sparse feature generator:
        \ref classias::sparse_feature_generator_base
- Classifiers and error functions
    - Linear binary classifier:
        \ref classias::classify::linear_binary
    - Linear binary classifier with logistic-sigmoid loss function:
        \ref classias::classify::linear_binary_logistic
    - Linear binary classifier with hinge loss function:
        \ref classias::classify::linear_binary_hinge
    - Linear multi classifier:
        \ref classias::classify::linear_multi
    - Linear multi classifier with soft-max function:
        \ref classias::classify::linear_binary_logistic
- Batch training algorithms
    - Gradient descent with L-BFGS/OW-LQN for logistic regression:
        \ref classias::train::lbfgs_logistic_binary
    - Gradient descent with L-BFGS/OW-LQN for multi-class logistic regression:
        \ref classias::train::lbfgs_logistic_multi
    - Scheduler for applying an online-training algorithm to a data set for binary classification:
        \ref classias::train::online_scheduler_binary
    - Scheduler for applying an online-training algorithm to a data set for multi classification:
        \ref classias::train::online_scheduler_multi
- Online training algorithms
    - Averaged perceptron for binary classification:
        \ref classias::train::averaged_perceptron_binary
    - Averaged perceptron for multi/candidate classification:
        \ref classias::train::averaged_perceptron_multi
    - Pegasos for binary classification:
        \ref classias::train::pegasos_binary
    - Pegasos for multi/candidate classification:
        \ref classias::train::pegasos_multi
    - Truncated gradient for binary classification:
        \ref classias::train::truncated_gradient_binary
    - Truncated gradient for multi/candidate classification:
        \ref classias::train::truncated_gradient_multi
- Basic data types
    - Instance weight:
        \ref classias::weight_base
    - Instance group number:
        \ref classias::group_base
    - Sparse vector:
        \ref classias::sparse_vector_base
    - Quark with one item (item-to-integer mapping):
        \ref classias::quark_base
    - Quark with two items (item-pair-to-integer mapping):
        \ref classias::quark2_base
    - Quark exception:
        \ref classias::quark_error
- Miscellaneous utilities
    - Accuracy counter:
        \ref classias::accuracy
    - Performance evaluation with precision, recall, and F1 scores
        \ref classias::precall
    - Parameter exchange
        \ref classias::parameter_exchange
    - Exception class for parameter exchange
        \ref classias::unknown_parameter
**/

/**
@defgroup sample_train_binary_batch Training a binary classifier on a data set
@{

@section Description
This code trains a model of binary classifier by using a data set read from
STDIN, and writes the resultant model to STDOUT. This code assumes that
features in the data set are represented by integer identifiers.

@include train_binary_batch.cpp

@}
*/

/**
@defgroup sample_train_binary_online Training a binary classifier with real online setting
@{

@section Description
This code trains a model of binary classifier with a real online setting,
i.e., the code reads each instance from STDIN, updates the model based on the
instance without storing it into an instance collection. This code assumes
that features in the data set are represented by integer identifiers.

@include train_binary_online.cpp

@}
*/

/**
@defgroup sample_tag_binary Binary classifier
@{

@section Description
This code reads a model of binary classifier from the file specified in the
first argument, reads a labeled or unlabeled data from STDIN, and writes the
predictions into STDOUT. If the input data is labeled, this code computes
the accuracy of the classification model.

@include tag_binary.cpp

@}
*/


#endif/*__CLASSIAS_CLASSIAS_H__*/
