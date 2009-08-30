/*
 *      Averaged perceptron.
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

#ifndef __CLASSIAS_TRAIN_AVERAGED_PERCEPTRON_H__
#define __CLASSIAS_TRAIN_AVERAGED_PERCEPTRON_H__

#include <iostream>

#include <classias/types.h>
#include <classias/classify/linear/binary.h>
#include <classias/classify/linear/multi.h>

namespace classias
{

namespace train
{

/**
 * The base class for Averaged Preceptron.
 *  This class implements internal variables, operations, and interface
 *  that are common for training a binary/multi classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 */
template <
    class error_tmpl
>
class averaged_perceptron_base
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of this class.
    typedef averaged_perceptron_base<error_tmpl> this_class;

protected:
    /// The array of feature weights (unaveraged or averaged).
    model_type m_w;
    /// The array of cumulative feature weights used for computing the average.
    model_type m_ws;
    /// The indicator whether m_w is averaged or not.
    bool m_averaged;

    /// The update count.
    int m_c;

    /// Parameter interface.
    parameter_exchange m_params;

public:
    /**
     * Constructs the object.
     */
    averaged_perceptron_base()
    {
        clear();
    }

    /**
     * Destructs the object.
     */
    virtual ~averaged_perceptron_base()
    {
    }

    /**
     * Resets the internal states and parameters to default.
     */
    void clear()
    {
        // Clear the weight vector.
        m_w.clear();
        m_ws.clear();
        this->initialize_weights();
    }

    /**
     * Sets the number of features.
     *  This function resizes the weight vector.
     *  @param  size        The number of features.
     */
    void set_num_features(size_t size)
    {
        m_w.resize(size);
        m_ws.resize(size);
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
        m_c = 1;
    }

    /**
     * Terminates a training process.
     *  This function performs a post-processing after a training process.
     */
    void finish()
    {
        this->average_weights();
    }

public:
    /**
     * Shows the copyright information.
     *  @param  os          The output stream.
     */
    void copyright(std::ostream& os)
    {
        os << "Averaged perceptron for " << error_type::name() << std::endl;
    }

    /**
     * Reports the current state of the training process.
     *  @param  os          The output stream.
     */
    void report(std::ostream& os)
    {
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
            m_ws[i] = 0.;
        }
        m_c = 1;
    }

    /**
     * Finalizes the weight vector.
     *  This function computes the actual weight vector W from the internal
     *  representation.
     */
    void average_weights()
    {
        if (!m_averaged) {
            for (size_t i = 0;i < m_w.size();++i) {
                m_w[i] -= m_ws[i] / m_c;
            }
            for (size_t i = 0;i < m_w.size();++i) {
                m_ws[i] = m_w[i];
            }
            m_averaged = true;
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
        this->average_weights();
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
 * Averaged Preceptron for binary classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 */
template <
    class error_tmpl
>
class averaged_perceptron_binary_base :
    public averaged_perceptron_base<error_tmpl>
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of the base class.
    typedef averaged_perceptron_base<error_tmpl> base_class;
    /// A synonym of this class.
    typedef averaged_perceptron_binary_base<error_tmpl> this_class;

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
        int& c = this->m_c;
        model_type& w = this->m_w;
        model_type& ws = this->m_ws;

        value_type loss = 0;
        error_type cls(w);
        cls.inner_product(it->begin(), it->end());
        if (static_cast<bool>(cls) != it->get_label()) {
            int y = static_cast<int>(it->get_label()) * 2 - 1;
            value_type delta = y * it->get_weight();
            update_weights(m_w, it->begin(), it->end(), delta);
            update_weights(m_ws, it->begin(), it->end(), c * delta);
            loss = 1;
        } else {
            loss = 0;
        }

        ++c;

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
     *  @param  w           The weight vector to which an update occurs.
     *  @param  first       The iterator pointing to the first element of
     *                      the feature vector.
     *  @param  last        The iterator pointing just beyond the last
     *                      element of the feature vector.
     *  @param  delta       The value to be added to the weights.
     */
    template <class iterator_type>
    inline void update_weights(
        model_type& w,
        iterator_type first,
        iterator_type last,
        value_type delta
        )
    {
        for (iterator_type it = first;it != last;++it) {
            w[it->first] += (delta * it->second);
        }
    }
};



/**
 * Averaged Preceptron for multi-class classification.
 *
 *  @param  error_tmpl  The type of the error (loss) function.
 *  @param  model_tmpl  The type of a weight vector for features.
 */
template <
    class error_tmpl
>
class averaged_perceptron_multi_base :
    public averaged_perceptron_base<error_tmpl>
{
public:
    /// The type implementing an error function.
    typedef error_tmpl error_type;
    /// The type implementing a model (weight vector for features).
    typedef typename error_type::model_type model_type;
    /// The type representing a value.
    typedef typename model_type::value_type value_type;
    /// A synonym of the base class.
    typedef averaged_perceptron_base<error_tmpl> base_class;
    /// A synonym of this class.
    typedef averaged_perceptron_multi_base<error_tmpl> this_class;

public:
    /**
     * Receives a training instance and updates feature weights.
     *  @param  it          An interator for the training instance.
     *  @param  fgen        The feature generator.
     *  @return value_type  The loss computed for the instance.
     */
    template <class iterator_type, class feature_generator_type>
    value_type update(iterator_type it, feature_generator_type& fgen)
    {
        const int L = (int)fgen.num_labels();

        // Define synonyms to avoid using "this->" for member variables.
        int& c = this->m_c;
        model_type& w = this->m_w;
        model_type& ws = this->m_ws;

        value_type loss = 0;
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

        if (cls.argmax() != it->get_label()) {
            int lr = it->get_label();
            int la = cls.argmax();

            update_weights(
                m_w,
                lr,
                fgen,
                it->attributes(lr).begin(),
                it->attributes(lr).end(),
                it->get_weight()
                );
            update_weights(
                m_w,
                lr,
                fgen,
                it->attributes(lr).begin(),
                it->attributes(lr).end(),
                c * it->get_weight()
                );

            update_weights(
                m_w,
                la,
                fgen,
                it->attributes(la).begin(),
                it->attributes(la).end(),
                -it->get_weight()
                );
            update_weights(
                m_w,
                la,
                fgen,
                it->attributes(la).begin(),
                it->attributes(la).end(),
                -c * it->get_weight()
                );
            loss = 1;
        } else {
            loss = 0;
        }

        ++c;
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
     *  @param  w           The weight vector to which an update occurs.
     *  @param  first       The iterator pointing to the first element of
     *                      the feature vector.
     *  @param  last        The iterator pointing just beyond the last
     *                      element of the feature vector.
     *  @param  delta       The value to be added to the weights.
     */
    template <class feature_generator_type, class iterator_type>
    inline void update_weights(
        model_type& w,
        int l,
        feature_generator_type& fgen,
        iterator_type first,
        iterator_type last,
        value_type delta
        )
    {
        for (iterator_type it = first;it != last;++it) {
            typename feature_generator_type::feature_type f =
                fgen.forward(it->first, l);
            if (0 <= f) {
                w[f] += (delta * it->second);
            }
        }
    }
};



typedef averaged_perceptron_binary_base<
    classify::linear_binary<weight_vector>
    > averaged_perceptron_binary;

typedef averaged_perceptron_multi_base<
    classify::linear_multi<weight_vector>
    > averaged_perceptron_multi;

};

};

#endif/*__CLASSIAS_TRAIN_AVERAGED_PERCEPTRON_BINARY_H__*/
