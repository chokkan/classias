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
#include <classias/parameters.h>

namespace classias
{

namespace train
{

/**
 * The base class for Trancated Gradient.
 *
 *  The detail of the algorithm is described in:
 *      John Langford, Lihong Li, and Tong Zhang.
 *      Sparse Online Learning via Truncated Gradient.
 *      JMLR 10(Mar):777-801, 2009.
 *
 *  This class implements internal variables, operations, and interface
 *  that are common for training a binary/multi classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 */
template <
    class error_tmpl
>
class truncated_gradient_base
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of this class.
    typedef truncated_gradient_base<error_tmpl> this_class;

    /// The type of progress information.
    struct report_type
    {
        /// The loss (the number of violations).
        value_type loss;
        /// The L1-norm of feature weights.
        value_type norm1;
        /// The L2-norm of feature weights.
        value_type norm2;
        /// The number of active features.
        int num_actives;

        void init()
        {
            loss = 0;
            norm1 = 0;
            norm2 = 0;
            num_actives = 0;
        }
    };
    report_type m_report;

protected:
    /// The array of feature weights.
    model_type m_w;
    /// The array of L1 penalties previously applied to weights.
    model_type m_penalty;

    /// The lambda (coefficient for L1 regularization).
    value_type m_lambda;
    /// The current learning rate.
    value_type m_eta;
    /// The offset of the update count.
    value_type m_t0;
    /// The update count.
    int m_t;
    /// The loss.
    value_type m_loss;
    /// The total amount of L1 penalty.
    value_type m_sum_penalty;

    /// Parameter interface.
    parameter_exchange m_params;
    /// The coefficient for L2 regularization.
    value_type m_c;
    /// The number of instances in the data set.
    value_type m_n;
    /// The initial learning rate.
    value_type m_eta0;
    /// The period for truncations.
    int m_truncate_period;
    /// The boolean value indicating whether m_w is truncated.
    bool m_truncated;

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
        m_params.init("c", &m_c, 1.,
            "Coefficient for L1 regularization.");
        m_params.init("n", &m_n, 1.,
            "The number of instances in the data set.");
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
        m_lambda = m_c / m_n;
        m_t = 0;
        m_t0 = 1.0 / (m_lambda * m_eta0);
        m_loss = 0;

        m_report.init();
    }

    /**
     * Terminates a training process.
     *  This function performs a post-processing after a training process.
     */
    void finish()
    {
        this->finalize_penalty(m_t, learning_rate(m_t));
        this->apply_penalty();
    }

    void discontinue()
    {
        this->apply_penalty();

        // Fill the progress information.
        m_report.init();
        m_report.loss = m_loss;
        for (size_t i = 0;i < m_w.size();++i) {
            value_type v = m_w[i];
            m_report.norm1 += std::fabs(v);
            m_report.norm2 += v * v;
            if (v != 0.) {
                ++m_report.num_actives;
            }
        }
        m_report.loss += m_c * m_report.norm1;

        // Reset the run-time information.
        m_loss = 0;
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
        os << "Loss: " << m_report.loss << std::endl;
        os << "Feature L1-norm: " << m_report.norm1 << std::endl;
        os << "Feature L2-norm: " << m_report.norm2 << std::endl;
        os << "Learning rate (eta): " << m_eta << std::endl;
        os << "Active features: " << m_report.num_actives << std::endl;
        os << "Total number of feature updates: " << m_t-1 << std::endl;
    }

protected:
    /**
     * Computes the learning rate for the update count.
     *  @param  t           The update count.
     *  @return value_type  The learning rate.
     */
    inline value_type learning_rate(int t)
    {
        return 1. / (m_lambda * (m_t0 + t));
    }

    /**
     * Accumulates the L1 penalty for the current update.
     *  @param  t           The update count.
     *  @param  eta         The learning rate for the update count.
     */
    inline void accumulate_penalty(int t, value_type eta)
    {
        if (t % m_truncate_period == 0) {
            m_sum_penalty += m_lambda * m_truncate_period * eta;
            m_truncated = false;
        }
    }

    /**
     * Finalizes the accmulation of L1 penalties.
     *  @param  t           The update count.
     *  @param  eta         The learning rate for the update count.
     */
    inline void finalize_penalty(int t, value_type eta)
    {
        m_sum_penalty += m_lambda * (t % m_truncate_period) * eta;
        m_truncated = false;
    }

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
        m_truncated = true;
    }

    /**
     * Applies the L1 penalties to the weight vector.
     */
    inline void apply_penalty()
    {
        if (!m_truncated) {
            for (int i = 0;i < (int)m_w.size();++i) {
                apply_penalty(i);
            }
            m_truncated = true;
        }
    }

    /**
     * Applies the L1 penalty to the weight of a feature.
     *  @param  i           The feature index.
     */
    inline void apply_penalty(int i)
    {
        value_type alpha = m_sum_penalty - m_penalty[i];
        if (0 < alpha) {
            if (0 < m_w[i]) {
                m_w[i] -= alpha;
                if (m_w[i] < 0) {
                    m_w[i] = 0;
                    m_penalty[i] = 0;
                    return;
                }
            } else if (m_w[i] < 0) {
                m_w[i] += alpha;
                if (0 < m_w[i]) {
                    m_w[i] = 0;
                    m_penalty[i] = 0;
                    return;
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

    value_type loss() const
    {
        return m_report.loss;
    }
};



/**
 * Truncated gradient for binary classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 */
template <
    class error_tmpl
>
class truncated_gradient_binary :
    public truncated_gradient_base<error_tmpl>
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of the base class.
    typedef truncated_gradient_base<error_tmpl> base_class;
    /// A synonym of this class.
    typedef truncated_gradient_binary<error_tmpl> this_class;

public:
    /**
     * Receives a training instance and updates feature weights.
     *  @param  it          An interator for the training instance.
     *  @return value_type  The loss computed for the instance.
     */
    template <class iterator_type>
    void update(iterator_type it)
    {
        // Synonyms to avoid "this->" for member variables in the base class.
        model_type& w = this->m_w;
        value_type& eta = this->m_eta;
        int& t = this->m_t;
        value_type& loss = this->m_loss;

        // Compute the learning rate for the current update.
        eta = this->learning_rate(++t);

        // Delay application of L1 penalties to the feature weights that
        // are relevant to the current instance.
        this->apply_penalty(it->begin(), it->end());

        // Compute the error and loss for the instance.
        error_type cls(w);
        cls.inner_product(it->begin(), it->end());
        value_type nlogp = 0.;
        value_type err = cls.error(it->get_label(), nlogp);
        loss += (it->get_weight() * nlogp);

        // Stochastic gradient descent without L1 regularization term.
        this->update_weights(
            it->begin(), it->end(), -err * eta * it->get_weight());

        // Accumulate the L1 penalty that should be applied in this update.
        this->accumulate_penalty(t, eta);
    }

    /**
     * Receives multiple training instances and updates feature weights.
     *  @param  first       The iterator pointing to the first instance.
     *  @param  last        The iterator pointing just beyond the last
     *                      instance.
     *  @return value_type  The loss computed for the instances.
     */
    template <class iterator_type>
    inline void update(iterator_type first, iterator_type last)
    {
        for (iterator_type it = first;it != last;++it) {
            this->update(it);
        }
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
            this->m_penalty[it->first] = this->m_sum_penalty;
        }
    }

    /**
     * Applies L1 penalties to the feature weights.
     *  This function applies L1 penalties to the weights in a feature vector.
     *  @param  first       The iterator pointing to the first element of
     *                      the feature vector.
     *  @param  last        The iterator pointing just beyond the last
     *                      element of the feature vector.
     */
    template <class iterator_type>
    inline void apply_penalty(iterator_type first, iterator_type last)
    {
        for (iterator_type it = first;it != last;++it) {
            base_class::apply_penalty(it->first);
        }
    }
};



/**
 * Truncated gradient for multi-class classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 */
template <
    class error_tmpl
>
class truncated_gradient_multi :
    public truncated_gradient_base<error_tmpl>
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of the base class.
    typedef truncated_gradient_base<error_tmpl> base_class;
    /// A synonym of this class.
    typedef truncated_gradient_multi<error_tmpl> this_class;

public:
    /**
     * Receives a training instance and updates feature weights.
     *  @param  it          An interator for the training instance.
     *  @param  fgen        The feature generator.
     *  @return value_type  The loss computed for the instance.
     */
    template <class iterator_type, class feature_generator_type>
    void update(iterator_type it, feature_generator_type& fgen)
    {
        const int L = (int)fgen.num_labels();

        // Synonyms to avoid "this->" for member variables in the base class.
        model_type& w = this->m_w;
        value_type& eta = this->m_eta;
        int& t = this->m_t;
        value_type& loss = this->m_loss;

        // Compute the learning rate for the current update.
        eta = this->learning_rate(++t);

        // Delay application of L1 penalties to the feature weights that
        // are relevant to the current instance.
        for (int i = 0;i < it->num_candidates(L);++i) {
            this->apply_penalty(
                i,
                fgen,
                it->attributes(i).begin(),
                it->attributes(i).end()
                );
        }

        // Compute the scores for the labels (candidates) in the instance.
        error_type cls(w);
        cls.resize(it->num_candidates(L));
        for (int i = 0;i < it->num_candidates(L);++i) {
            cls.inner_product(
                i,
                fgen,
                it->attributes(i).begin(),
                it->attributes(i).end(),
                i
                );
        }
        cls.finalize();

        // Compute the loss for the instance.
        loss += -it->get_weight() * cls.logprob(it->get_label());

        // Updates the feature weights.
        value_type gain = eta * it->get_weight();
        for (int i = 0;i < it->num_candidates(L);++i) {
            // Computes the error for the label (candidate).
            value_type err = cls.error(i, it->get_label());

            // Update the feature weights.
            update_weights(
                i,
                fgen,
                it->attributes(i).begin(),
                it->attributes(i).end(),
                -err * gain 
                );
        }

        // Accumulate the L1 penalty that should be applied in this update.
        this->accumulate_penalty(t, eta);
    }

    /**
     * Receives multiple training instances and updates feature weights.
     *  @param  first       The iterator pointing to the first instance.
     *  @param  last        The iterator pointing just beyond the last
     *                      instance.
     *  @return value_type  The loss computed for the instances.
     */
    template <class iterator_type>
    inline void update(iterator_type first, iterator_type last)
    {
        for (iterator_type it = first;it != last;++it) {
            this->update(it);
        }
    }

protected:
    /**
     * Adds a value to weights associated with a feature vector.
     *  @param  l           The candidate index.
     *  @param  fgen        The feature generator.
     *  @param  first       The iterator pointing to the first element of
     *                      the feature vector.
     *  @param  last        The iterator pointing just beyond the last
     *                      element of the feature vector.
     *  @param  delta       The value to be added to the weights.
     */
    template <class feature_generator_type, class iterator_type>
    inline void update_weights(
        int l,
        feature_generator_type& fgen,
        iterator_type first,
        iterator_type last,
        value_type delta
        )
    {
        for (iterator_type it = first;it != last;++it) {
            typename feature_generator_type::feature_type f;
            if (fgen.forward(it->first, l, f)) {
                this->m_w[f] += delta * it->second;
                this->m_penalty[f] = this->m_sum_penalty;
            }
        }
    }

    /**
     * Applies L1 penalties to the feature weights.
     *  This function applies L1 penalties to the weights in a feature vector.
     *  @param  l           The candidate index.
     *  @param  fgen        The feature generator.
     *  @param  first       The iterator pointing to the first element of
     *                      the feature vector.
     *  @param  last        The iterator pointing just beyond the last
     *                      element of the feature vector.
     */
    template <class feature_generator_type, class iterator_type>
    inline void apply_penalty(
        int l,
        feature_generator_type& fgen,
        iterator_type first,
        iterator_type last
        )
    {
        for (iterator_type it = first;it != last;++it) {
            typename feature_generator_type::feature_type f;
            if (fgen.forward(it->first, l, f)) {
                base_class::apply_penalty(f);
            }
        }
    }
};

};

};

/*

The detail of this algorithm is described in:

John Langford, Lihong Li, and Tong Zhang.
Sparse Online Learning via Truncated Gradient.
JMLR 10(Mar):777-801, 2009.

This is the pseudo-code of a naive implementation of the truncated gradient.

def apply_penalty(W, i, alpha):
    if 0 < W[i]:
        W[i] -= alpha
        if W[i] < 0:
            W[i] == 0
    elif W[i] < 0:
        W[i] += alpha
        if 0 < W[i]:
            W[i] = 0

def train(data, lambda, max_epoch, truncate_period):
    W = 0
    t = 1
    eta = 1
    for epoch in range(max_epoch):
        for inst in data:
            eta = 1.0 / (lambda * t)
            err = error_function(W, inst)
            for f, v in inst:
                W[f] -= err * eta * v
            if t % truncate_period == 0:
                alpha = lambda * truncate_period * eta
                for i in range(len(W)):
                    apply_penalty(W, i, alpha)
            t += 1
    alpha = lambda * (t % truncate_period) * eta
    for i in range(len(W)):
        apply_penalty(W, i, alpha)
    return W

This pseudo-code has O(|W|) computation to apply L1 penalties for each update.
In order to reduce this computational cost, we introduces a variable
(cumulative L1 penalty) that sums up L1 penalties so far and a vector P whose
element P[i] stores the value of the cumulative L1 penalty when the feature
weight W[i] is updated last time. By using these variables, we can perform a
delay application of L1 penalties ((cumulative L1 penalty) - P[i]) until
a feature weight is used for computing the error of a given instance.

This is the final form of the efficient implementation.

def train(data, lambda, max_epoch, truncate_period):
    W = 0
    P = 0
    t = 1
    s = 0
    eta = 1
    for epoch in range(max_epoch):
        for inst in data:
            eta = 1.0 / (lambda * t)
            for f, v, in inst:
                apply_penalty(W, f, s - P[f])
                P[f] = s
            err = error_function(W, inst)
            for f, v in inst:
                W[f] -= err * eta * v
            if t % truncate_period == 0:
                s += lambda * truncate_period * eta
            t += 1
    for i in range(len(W)):
        apply_penalty(W, i, s - P[f])
    return W

*/

#endif/*__CLASSIAS_TRAIN_PEGASOS_H__*/
