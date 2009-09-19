/*
 *      Primal Estimated sub-GrAdient SOlver (Pegasos).
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

#ifndef __CLASSIAS_TRAIN_PEGASOS_H__
#define __CLASSIAS_TRAIN_PEGASOS_H__

#include <iostream>

#include <classias/types.h>
#include <classias/parameters.h>

namespace classias
{

namespace train
{

/**
 * The base class for Primal Estimated sub-GrAdient SOlver (Pegasos).
 *  The detail of this algorithm is described in:
 *
 *  -   Shai Shalev-Shwartz, Yoram Singer, and Nathan Srebro.
 *      Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
 *      In Proc. of ICML 2007, pp 807-814, 2007.
 *
 *  This class implements internal variables, operations, and interface
 *  that are common for training a binary/multi classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 *  @param  model_tmpl  The type of a weight vector for features.
 */
template <
    class error_tmpl
>
class pegasos_base
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of this class.
    typedef pegasos_base<error_tmpl> this_class;

    /// The type of progress information.
    struct report_type
    {
        /// The loss (the number of violations).
        value_type loss;
        /// The L2-norm of feature weights.
        value_type norm2;
    };
    report_type m_report;

protected:
    /// The array of feature weights.
    model_type m_model;

    /// The lambda (coefficient for L2 regularization).
    value_type m_lambda;
    /// The square of the L2-norm of feature weights.
    value_type m_norm22;
    /// The decay factor for feature weights.
    value_type m_decay;
    /// The projection factor for feature weights.
    value_type m_proj;
    /// The scaling factor for feature weights.
    value_type m_scale;
    /// The current learning rate.
    value_type m_eta;
    /// The offset of the update count.
    value_type m_t0;
    /// The loss.
    value_type m_loss;
    /// The update count.
    int m_t;

    /// Parameter interface.
    parameter_exchange m_params;
    /// The coefficient for L2 regularization.
    value_type m_c;
    /// The number of instances in the data set.
    value_type m_n;
    /// The initial learning rate.
    value_type m_eta0;

public:
    /**
     * Constructs the object.
     */
    pegasos_base()
    {
        clear();
    }

    /**
     * Destructs the object.
     */
    virtual ~pegasos_base()
    {
    }

    /**
     * Resets the internal states and parameters to default.
     */
    void clear()
    {
        // Clear the weight vector.
        m_model.clear();
        this->initialize_weights();

        // Initialize the parameters.
        m_params.init("c", &m_c, 1.,
            "Coefficient for L2 regularization.");
        m_params.init("n", &m_n, 1.,
            "The number of instances in the data set.");
        m_params.init("eta", &m_eta0, 0.1,
            "Initial learning rate");
    }

    /**
     * Sets the number of features.
     *  This function resizes the weight vector.
     *  @param  size        The number of features.
     */
    void set_num_features(size_t size)
    {
        m_model.resize(size);
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

        m_report.loss = 0;
        m_report.norm2 = 0;
    }

    /**
     * Terminates a training process.
     *  This function performs a post-processing after a training process.
     */
    void finish()
    {
        this->rescale_weights();
    }

    void discontinue()
    {
        this->rescale_weights();

        // Fill the progress information.
        m_report.loss = m_loss;
        m_report.norm2 = std::sqrt(m_norm22);

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
        os << "Pegasos for " << error_type::name() << std::endl;
    }

    /**
     * Reports the current state of the training process.
     *  @param  os          The output stream.
     */
    void report(std::ostream& os)
    {
        os << "Loss: " << m_report.loss << std::endl;
        os << "Feature L2-norm: " << m_report.norm2 << std::endl;
        os << "Learning rate (eta): " << m_eta << std::endl;
        os << "Total number of feature updates: " << m_t << std::endl;
    }

protected:
    /**
     * Initializes the weight vector.
     *  This function sets W = 0.
     */
    void initialize_weights()
    {
        for (size_t i = 0;i < m_model.size();++i) {
            m_model[i] = 0.;
        }
        m_norm22 = 0;
        m_decay = 1;
        m_proj = 1;
        m_scale = 1;
    }

    /**
     * Finalizes the weight vector.
     *  This function computes the actual weight vector W from the internal
     *  representation (V, decay, proj).
     */
    void rescale_weights()
    {
        m_norm22 = 0;
        for (size_t i = 0;i < m_model.size();++i) {
            m_model[i] *= m_scale;
            m_norm22 += (m_model[i] * m_model[i]);
        }

        m_decay = 1;
        m_proj = 1;
        m_scale = 1;
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
        if (m_scale != 1.) {
            this->rescale_weights();
        }
        return m_model;
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
 * Pegasos for binary classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 */
template <
    class error_tmpl
>
class pegasos_binary :
    public pegasos_base<error_tmpl>
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of the base class.
    typedef pegasos_base<error_tmpl> base_class;
    /// A synonym of this class.
    typedef pegasos_binary<error_tmpl> this_class;

public:
    /**
     * Receives a training instance and updates feature weights.
     *  @param  it          An interator for the training instance.
     *  @return value_type  The loss computed for the instance.
     */
    template <class iterator_type>
    void update(iterator_type it)
    {
        // Define synonyms to avoid using "this->" for member variables.
        model_type& model = this->m_model;
        value_type& eta = this->m_eta;
        value_type& lambda = this->m_lambda;
        value_type& decay = this->m_decay;
        value_type& proj = this->m_proj;
        value_type& scale = this->m_scale;
        value_type& norm22 = this->m_norm22;
        int& t = this->m_t;
        value_type& t0 = this->m_t0;
        value_type& loss = this->m_loss;

        // Learning rate: eta = 1. / (lambda * (t0 + t)).
        eta = 1. / (lambda * (t0 + t));

        // Compute the error for the instance.
        value_type nlogp = 0.;
        error_type cls(model);
        cls.inner_product(it->begin(), it->end());
        cls.scale(scale);
        value_type err = cls.error(it->get_label(), nlogp);
        loss += (it->get_weight() * nlogp);

        // W *= (1 - eta * lambda), equivalent to L2 regularization.
        // Insted of applying the decay factor to the weight vector,
        // let W = (decay * proj) * V and remember the products of
        // decay factors. This avoids an O(K) computation for every
        // updates.
        decay *= (1. - eta * lambda);
        scale = decay * proj;

        // W -= (err * eta * x) <==> V -= (err * eta * x) / (decay * proj).
        // Thus, gain = eta / (decay * proj).
        value_type gain = 1;
        if (0 < decay) {
            gain = eta / scale;
        } else {
            // decay = 0 implies that W should be initialized to 0.
            this->initialize_weights();
            gain = 1;
        }

        // Update the feature weights.
        update_weights(it->begin(), it->end(), -gain * err * it->get_weight());

        // Project the weight vector within an L2 ball.
        if (1 < lambda * norm22 * scale * scale) {
            proj = 1.0 / (sqrt(lambda * norm22) * scale);
        }

        // Increment the update count.
        ++t;
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
        model_type& model = this->m_model;
        value_type& norm22 = this->m_norm22;

        for (iterator_type it = first;it != last;++it) {
            value_type w = model[it->first];
            value_type d = delta * it->second;
            model[it->first] += d;
            norm22 += d * (d + w + w);
        }
    }
};



/**
 * Pegasos for multi classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 *  @param  model_tmpl  The type of a weight vector for features.
 */
template <
    class error_tmpl
>
class pegasos_multi :
    public pegasos_base<error_tmpl>
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of the base class.
    typedef pegasos_base<error_tmpl> base_class;
    /// A synonym of this class.
    typedef pegasos_binary<error_tmpl> this_class;

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

        // Define synonyms to avoid using "this->" for member variables.
        model_type& model = this->m_model;
        value_type& eta = this->m_eta;
        value_type& lambda = this->m_lambda;
        value_type& decay = this->m_decay;
        value_type& proj = this->m_proj;
        value_type& scale = this->m_scale;
        value_type& norm22 = this->m_norm22;
        int& t = this->m_t;
        value_type& t0 = this->m_t0;
        value_type& loss = this->m_loss;

        // Learning rate: eta = 1. / (lambda * (t0 + t)).
        eta = 1. / (lambda * (t0 + t));

        // Compute the scores for the labels (candidates) in the instance.
        value_type nlogp = 0.;
        error_type cls(model);
        cls.resize(it->num_candidates(L));
        for (int i = 0;i < it->num_candidates(L);++i) {
            cls.inner_product(
                i,
                fgen,
                it->attributes(i).begin(),
                it->attributes(i).end(),
                i
                );
            cls.scale(i, scale);
        }
        cls.finalize();

        // Compute the loss for the instance.
        loss += -it->get_weight() * cls.logprob(it->get_label());

        // W *= (1 - eta * lambda), equivalent to L2 regularization.
        // Insted of applying the decay factor to the weight vector,
        // let W = (decay * proj) * V and remember the products of
        // decay factors. This avoids an O(K) computation for every
        // updates.
        decay *= (1. - eta * lambda);
        scale = decay * proj;

        // W -= (err * eta * x) <==> V -= (err * eta * x) / (decay * proj).
        // Thus, gain = eta / (decay * proj).
        value_type gain = 1;
        if (0 < decay) {
            gain = eta / scale;
        } else {
            // decay = 0 implies that W should be initialized to 0.
            this->initialize_weights();
            gain = 1;
        }
        gain *= it->get_weight();

        // Updates the feature weights.
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


        // Project the weight vector within an L2 ball.
        if (1 < lambda * norm22 * scale * scale) {
            proj = 1.0 / (sqrt(lambda * norm22) * scale);
        }

        // Increment the update count.
        ++t;
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
     *  @param  l           The label or candidate index.
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
        model_type& model = this->m_model;
        value_type& norm22 = this->m_norm22;

        for (iterator_type it = first;it != last;++it) {
            typename feature_generator_type::feature_type f;
            if (fgen.forward(it->first, l, f)) {
                value_type w = model[f];
                value_type d = delta * it->second;
                model[f] += d;
                norm22 += d * (d + w + w);
            }
        }
    }
};

};

};

/*

The detail of this algorithm is described in:

Shai Shalev-Shwartz, Yoram Singer, and Nathan Srebro.
Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
In Proc. of ICML 2007, pp 807-814, 2007.

This is a naive pseudo-code of the Pegasos algorithm.

1:  W = 0
2:  for epoch in range(max_epoch):
3:      for inst in data:
4:          eta = 1.0 / (lambda * t)
5:          err = error_function(W, inst)
6:          W = (1 - eta * lambda) * W
7:          for f, v in inst:
8:              W[f] -= err * eta * v
9:          if 1 / sqrt(lambda) <= norm2(W):
10:             W /= (sqrt(lambda) * norm2(W))
11:         t += 1
12: return W

Step 8 requires O(d) computation, where d is the number of active features
in the current instance. Steps 6 and 10 require O(k) computations, where k
is the total number of features. When the training data is high-dimensional
but sparse, i.e., d << k, the above code is very inefficient.

In order to avoid O(k) computations in Steps 6 and 10, we maintain two
scalar values, decay and proj, and replace the original weight vector W:
    W = (decay * proj * V).
Instead of scaling the actual weight vector W, we update the amount of scaling
factors (decay and proj) that have been applied so far. Because of this change
to the weight vector W, we also need to modify Step 8 so that:
    W -= (err * eta * x) <===> V -= (err * eta * x) / (decay * proj).
This idea of efficient implementation is proposed in the original paper.

We also maintain norm22 (the square of the L2-norm of the weight vector W)
whenever an element of the vector is updated. When delta is added to W[i],
the change of the norm22 is (2 * W[i] * delta + delta^2) because:
    |W[i] + delta|^2 = W[i]^2 + 2 * W[i] * delta + delta^2

This is the final version of the efficient implementation:

1a: V = 0
1b: norm22 = 0
1c: decay = 1
1d: proj = 1
2:  for epoch in range(max_epoch):
3:      for inst in data:
4a:         eta = 1.0 / (lambda * t)
4b:         scale = decay * proj
5:          err = error_function(scale * V, inst)
6a:         decay *= (1 - eta * lambda)
6b:         if decay > 0:
6c:             gain = err / decay
6d:         else:
6e:             V = 0
6f:             norm22 = 0
6g:             decay = 1
6h:             proj = 1
6i:             gain = 1
7:          for f, v in inst:
8a:             delta = -gain * eta * v
8b:             V[f] += delta
8c:             norm22 += delta * (delta + V[f] + V[f])
9:          if 1 <= lambda * norm22 * scale * scale:
10:             proj = 1.0 / (sqrt(lambda * norm22) * scale)
11:         t += 1
12: return (scale * V)

*/

#endif/*__CLASSIAS_TRAIN_PEGASOS_H__*/
