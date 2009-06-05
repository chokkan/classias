/*
 *		Trainer for candidate classifier using L-BFGS.
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

#ifndef __CLASSIAS_TRAIN_LBFGS_CANDIDATE_H__
#define __CLASSIAS_TRAIN_LBFGS_CANDIDATE_H__

#include <float.h>
#include <cmath>
#include <ctime>
#include <iostream>

#include "base.h"
#include <classias/classify/linear/candidate.h>
#include <classias/evaluation.h>

namespace classias
{

/**
 * Training a log-linear model using the maximum entropy modeling.
 *  @param  data_tmpl           Training data class.
 *  @param  value_tmpl          The type for computation.
 */
template <
    class data_tmpl,
    class value_tmpl = double
>
class trainer_lbfgs_candidate : public lbfgs_base<value_tmpl>
{
protected:
    /// A type representing a data set for training.
    typedef data_tmpl data_type;
    /// A type representing values for internal computations.
    typedef value_tmpl value_type;
    /// A synonym of the base class.
    typedef lbfgs_base<value_tmpl> base_class;
    /// A synonym of this class.
    typedef trainer_lbfgs_candidate<data_type, value_type> this_class;
    /// A type representing an instance in the training data.
    typedef typename data_type::instance_type instance_type;
    /// A type providing a read-only random-access iterator for instances.
    typedef typename data_type::const_iterator const_iterator;
    /// A type representing a candidate in an instance.
    typedef typename instance_type::candidate_type candidate_type;
    /// A type representing an attribute in a candidate.
    typedef typename data_type::attribute_type attribute_type;
    /// A type representing a label.
    typedef typename data_type::label_type label_type;
    /// The type of a classifier.
    typedef classify::linear_candidate_logistic<attribute_type, label_type, value_type, value_type const*> classifier_type;

    /// An array [K] of observation expectations.
    value_type *m_oexps;

    /// A data set for training.
    const data_type* m_data;

public:
    trainer_lbfgs_candidate()
    {
        m_oexps = NULL;
        m_data = NULL;
        clear();
    }

    virtual ~trainer_lbfgs_candidate()
    {
        clear();
    }

    void clear()
    {
        delete[] m_oexps;
        m_oexps = NULL;

        m_data = NULL;
        base_class::clear();
    }

    virtual value_type loss_and_gradient(
        const value_type *x,
        value_type *g,
        const int n
        )
    {
        value_type loss = 0;
        const data_type& data = *m_data;
        classifier_type cls(x);

        // Initialize the gradients with (the negative of) observation expexcations.
        for (int i = 0;i < n;++i) {
            g[i] = -m_oexps[i];
        }

        // For each instance in the data.
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            int i, itrue = -1;
            const instance_type& inst = *iti;

            // Exclude instances for holdout evaluation.
            if (inst.get_group() == this->m_holdout) {
                continue;
            }

            // Compute a prob[i] for each candidate #i.
            cls.resize(inst.size());
            for (i = 0;i < (int)inst.size();++i) {
                const candidate_type& cand = inst[i];
                cls.inner_product(i, cand.begin(), cand.end());
                if (cand.get_truth()) {
                    itrue = i;
                }
            }
            cls.finalize();

            // Ignore an instance with no true candidate.
            if (itrue == -1) {
                continue;
            }

            // Accumulate the model expectations of features.
            for (i = 0;i < (int)inst.size();++i) {
                const candidate_type& cand = inst[i];
                typename candidate_type::const_iterator it;
                for (it = cand.begin();it != cand.end();++it) {
                    g[it->first] += cls.prob(i) * it->second;
                }
            }

            // Accumulate the loss for predicting the instance.
            loss -= std::log(cls.prob(itrue));
        }

        return loss;
    }

    int train(
        const data_type& data,
        std::ostream& os,
        int holdout = -1
        )
    {
        const size_t K = data.num_features();
        const size_t L = data.num_labels();

        // Initialize feature expectations and weights.
        this->initialize_weights(K);
        m_oexps = new double[K];
        for (size_t k = 0;k < K;++k) {
            m_oexps[k] = 0.;
        }

        // Report the training parameters.
        os << "Multi-candidate logistic regression using L-BFGS" << std::endl;
        this->m_params.show(os);
        os << "lbfgs.regularization_start: " << data.get_user_feature_start() << std::endl;
        os << std::endl;

        // Compute observation expectations of the features.
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            const instance_type& inst = *iti;

            // Skip instances for holdout evaluation.
            if (inst.get_group() == holdout) {
                continue;
            }

            // Compute the observation expectations.
            for (int i = 0;i < (int)inst.size();++i) {
                const candidate_type& cand = inst[i];
                if (cand.get_truth()) {
                    typename candidate_type::const_iterator it;
                    for (it = cand.begin();it != cand.end();++it) {
                        m_oexps[it->first] += it->second;
                    }
                }
            }
        }

        // Call the L-BFGS solver.
        m_data = &data;
        int ret = lbfgs_solve(
            (const int)K,
            os,
            holdout,
            data.get_user_feature_start()
            );

        // Report the result from the L-BFGS solver.
        this->lbfgs_output_status(os, ret);
        return ret;
    }

    void holdout_evaluation()
    {
        std::ostream& os = *(this->m_os);
        const data_type& data = *(this->m_data);
        const value_type *x = this->m_weights;
        classifier_type cls(x);
        accuracy acc;
        confusion_matrix matrix(data.labels.size());

        // Loop over instances.
        for (const_iterator iti = data.begin();iti != data.end();++iti) {
            int itrue = -1;
            const instance_type& inst = *iti;

            // Exclude instances for holdout evaluation.
            if (inst.get_group() != this->m_holdout) {
                continue;
            }

            // Compute the probability prob[i] for each candidate #i.
            cls.resize(inst.size());
            for (int i = 0;i < (int)inst.size();++i) {
                const candidate_type& cand = inst[i];
                cls.inner_product(i, cand.begin(), cand.end());
                if (cand.get_truth()) {
                    itrue = i;
                }
            }
            cls.finalize();

            int imax = cls.argmax();
            acc.set(itrue == imax);
            matrix.set(inst[itrue].get_label(), inst[imax].get_label());
        }

        // Report accuracy, precision, recall, and f1 score.
        acc.output(os);
        matrix.output_micro(os, data.positive_labels.begin(), data.positive_labels.end());
        matrix.output_macro(os, data.positive_labels.begin(), data.positive_labels.end());
    }
};

};

#endif/*__CLASSIAS_TRAIN_LBFGS_CANDIDATE_H__*/
