/*
 *      SGD with Trancated Gradient for L1 regularization.
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

#ifndef __CLASSIAS_TRAIN_TRUNCATED_GRADIENT_H__
#define __CLASSIAS_TRAIN_TRUNCATED_GRADIENT_H__

#include <iostream>

#include <classias/types.h>
#include <classias/classify/linear/binary.h>
#include <classias/classify/linear/multi.h>

namespace classias
{

namespace train
{

/**
 * The base class for Trancated Gradient.
 *  This class implements internal variables, operations, and interface
 *  that are common for training a binary/multi classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 *  @param  model_tmpl  The type of a weight vector for features.
 */
template <
    class error_tmpl,
    class model_tmpl
>
class truncated_gradient_base
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef model_tmpl model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of this class.
    typedef truncated_gradient_base<error_tmpl, model_tmpl> this_class;

protected:
    /// The array of feature weights.
    model_type m_w;
    /// The array of L1 penalties previously applied to weights.
    model_type m_penalty;

    /// The current learning rate.
    value_type m_eta;
    /// The offset of the update count.
    value_type m_t0;
    /// The update count.
    int m_t;
    /// The total amount of L1 penalty.
    value_type m_sum_penalty;

    /// Parameter interface.
    parameter_exchange m_params;
    /// The lambda (coefficient for L2 regularization).
    value_type m_lambda;
    /// The initial learning rate.
    value_type m_eta0;
    /// The period 
    int m_truncate_period;
    bool m_trucated;

public:
    /**
     * Constructs the object.
     */
    truncated_gradient_base()
    {
        clear();
    }

    /**
     * Destructs the object.
     */
    virtual ~truncated_gradient_base()
    {
    }

    /**
     * Resets the internal states and parameters to default.
     */
    void clear()
    {
        // Clear the weight vector.
        m_w.clear();
        m_penalty.clear();
        this->initialize_weights();

        // Initialize the parameters.
        m_params.init("lambda", &m_lambda, 0.001,
            "Coefficient (lambda) for L1-regularization.");
        m_params.init("eta", &m_eta0, 0.1,
            "Initial learning rate");
        m_params.init("truncate_period", &m_truncate_period, 1,
            "Period for truncate");
    }

    /**
     * Sets the number of features.
     *  This function resizes the weight vector.
     *  @param  size        The number of features.
     */
    void set_num_features(size_t size)
    {
        m_w.resize(size);
        m_penalty.resize(size);
        this->initialize_weights();
    }

public:
    /**
     * Starts a training process.
     *  This function resets the internal states, and prepares for a training
     *  process.
     */
    void start()
    {
        this->initialize_weights();
        m_t = 0;
        m_t0 = 1.0 / (m_lambda * m_eta0);
    }

    /**
     * Terminates a training process.
     *  This function performs a post-processing after a training process.
     */
    void finish()
    {
        this->apply_penalty();
    }

public:
    /**
     * Shows the copyright information.
     *  @param  os          The output stream.
     */
    void copyright(std::ostream& os)
    {
        os << "Truncated Gradient for " << error_type::name() << std::endl;
    }

    /**
     * Reports the current state of the training process.
     *  @param  os          The output stream.
     */
    void report(std::ostream& os)
    {
        // Count the number of active features.
        int num_active_features = 0;
        for (int i = 0;i < (int)m_w.size();++i) {
            value_type alpha = m_sum_penalty - m_penalty[i];
            if (m_w[i] < -alpha || alpha < m_w[i]) {
                ++num_active_features;
            }
        }

        os << "Learning rate (eta): " << m_eta << std::endl;
        os << "Active features: " << num_active_features << std::endl;
        os << "Total number of feature updates: " << m_t-1 << std::endl;
    }

protected:
    /**
     * Initializes the weight vector.
     *  This function sets W = 0.
     */
    void initialize_weights()
    {
        for (size_t i = 0;i < m_w.size();++i) {
            m_w[i] = 0.;
            m_penalty[i] = 0.;
        }
        m_sum_penalty = 0.;
        m_trucated = false;
    }

    inline void apply_penalty()
    {
        for (int i = 0;i < (int)m_w.size();++i) {
            apply_penalty(i);
        }
    }

    inline void apply_penalty(int i)
    {
        value_type alpha = m_sum_penalty - m_penalty[i];
        if (0 < alpha) {
            if (0 < m_w[i]) {
                m_w[i] -= alpha;
                if (m_w[i] < 0) {
                    m_w[i] = 0;
                }
            } else if (m_w[i] < 0) {
                m_w[i] += alpha;
                if (0 < m_w[i]) {
                    m_w[i] = 0;
                }
            }
            m_penalty[i] = m_sum_penalty;
        }
    }

public:
    /**
     * Obtains the parameter interface.
     *  @return parameter_exchange& The parameter interface associated with
     *                              this algorithm.
     */
    parameter_exchange& params()
    {
        return m_params;
    }

public:
    /**
     * Obtains an access to the weight vector (model).
     *  @return model_type&         The weight vector (model).
     */
    model_type& model()
    {
        this->apply_penalty();
        return m_w;
    }

    /**
     * Obtains a read-only access to the weight vector (model).
     *  @return const model_type&   The weight vector (model).
     */
    const model_type& model() const
    {
        // Force to remove the const modifier for rescaling.
        return const_cast<this_class*>(this)->model();
    }
};



/**
 * Truncated gradient for binary classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 *  @param  model_tmpl  The type of a weight vector for features.
 */
template <
    class error_tmpl,
    class model_tmpl
>
class truncated_gradient_binary :
    public truncated_gradient_base<error_tmpl, model_tmpl>
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef model_tmpl model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of the base class.
    typedef truncated_gradient_base<error_tmpl, model_tmpl> base_class;
    /// A synonym of this class.
    typedef truncated_gradient_base<error_tmpl, model_tmpl> this_class;

public:
    /**
     * Receives a training instance and updates feature weights.
     *  @param  it          An interator for the training instance.
     *  @return value_type  The loss computed for the instance.
     */
    template <class iterator_type>
    value_type update(iterator_type it)
    {
        // Define synonyms to avoid using "this->" for member variables.
        model_type& w = this->m_w;
        model_type& penalty = this->m_penalty;
        value_type& eta = this->m_eta;
        value_type& lambda = this->m_lambda;
        value_type& sum_penalty = this->m_sum_penalty;
        int& truncate_period = this->m_truncate_period;
        int& t = this->m_t;
        value_type& t0 = this->m_t0;

        // Learning rate: eta = 1. / (lambda * (t0 + t)).
        ++t;
        eta = 1. / (lambda * (t0 + t));

        // Apply 
        apply_penalty(it->begin(), it->end());

        // Compute the error for the instance.
        value_type nlogp = 0.;
        error_type cls(w);
        cls.inner_product(it->begin(), it->end());
        value_type err = cls.error(it->get_label(), nlogp);
        value_type loss = (it->get_weight() * nlogp);

        // Update the feature weights.
        update_weights(it->begin(), it->end(), -err * eta * it->get_weight());

        //
        if (t % truncate_period == 0) {
            sum_penalty += lambda * truncate_period * eta;
        }

        return loss;
    }

    /**
     * Receives multiple training instances and updates feature weights.
     *  @param  first       The iterator pointing to the first instance.
     *  @param  last        The iterator pointing just beyond the last
     *                      instance.
     *  @return value_type  The loss computed for the instances.
     */
    template <class iterator_type>
    inline value_type update(iterator_type first, iterator_type last)
    {
        value_type loss = 0;
        for (iterator_type it = first;it != last;++it) {
            loss += this->update(it);
        }
        return loss;
    }

protected:
    /**
     * Adds a value to weights associated with a feature vector.
     *  @param  first       The iterator pointing to the first element of
     *                      the feature vector.
     *  @param  last        The iterator pointing just beyond the last
     *                      element of the feature vector.
     *  @param  delta       The value to be added to the weights.
     */
    template <class iterator_type>
    inline void update_weights(iterator_type first, iterator_type last, value_type delta)
    {
        for (iterator_type it = first;it != last;++it) {
            this->m_w[it->first] += delta * it->second;
        }
    }

    template <class iterator_type>
    inline void apply_penalty(iterator_type first, iterator_type last)
    {
        for (iterator_type it = first;it != last;++it) {
            base_class::apply_penalty(it->first);
        }
    }
};



/** Truncate gradient for binary classification with logistic loss. */
typedef truncated_gradient_binary<
    classify::linear_binary_logistic<int, double, weight_vector>,
    weight_vector
    > truncated_gradient_binary_logistic_loss;

};

};

/*

This is a pseudo-code of the naive implementation of the truncated gradient.

1:  W = 0; t = 1
2:  for epoch in range(max_epoch):
3:      for inst in data:
4:          eta = 1.0 / (lambda * t)
5:          err = error_function(W, inst)
6:          for f, v in inst:
7:              W[f] -= err * eta * v
8:          if t % K == 0:
9:              alpha = g * K * eta
10:             for i in range(len(W)):
11:                 if 0 < W[i] <= theta:
12:                     W[i] -= alpha
13:                     if W[i] < 0:
14:                         W[i] = 0
15:                 elif -theta <= W[i] < 0:
16:                     W[i] += alpha
17:                     if W[i] > 0:
18:                         W[i] = 0
19:         t += 1
20: return W

I think setting theta < \infty is not a good idea.

W = 0
P = 0
t = 1
s = 0
for epoch in range(max_epoch):
    for inst in data:
        eta = 1.0 / (lambda * t)
        for f, v in inst:
            alpha = s - P[f]
            if 0 < W[f]:
                W[f] -= alpha
                if W[f] < 0:
                    W[f] = 0
            else:
                W[f] += alpha
                if W[f] > 0:
                    W[f] = 0
            P[f] = s
        err = error_function(W, inst)
        for f, v in inst:
            W[f] -= err * eta * v
        if t % K == 0:
            s += g * K * eta
        t += 1
return W


*/

#endif/*__CLASSIAS_TRAIN_PEGASOS_H__*/
