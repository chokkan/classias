/*
 *		Binary classifier with linear models.
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

#ifndef __CLASSIAS_CLASSIFY_LINEAR_BINARY_H__
#define __CLASSIAS_CLASSIFY_LINEAR_BINARY_H__

#include <cmath>

namespace classias
{

namespace classify
{

/**
 * A template class for linear binary classifier.
 *
 *  This template class implements a binary classifier with a linear
 *  discriminant model. The types of features and models are customizable.
 *  Any type that yields a feature weight with operator \c [] is acceptable
 *  for a model. The type of features must be compatible with the argument
 *  type of the operator \c [] of the model. The typical configurations are:
 *      - a model is implemented by \c std::vector<double>, and features are
 *        integers (indices to the model array).
 *      - a model is implemented by \c std::map<std::string, double>, and
 *        features are strings.
 *  
 *  The former configuration is preferable for faster training. The latter
 *  may be useful for implementing a classifier with string features.
 *
 *  @param  model_tmpl      The type of a model (container of feature weights).
 *                          Any type that yields a feature weight with
 *                          operator \c [] is usable (e.g., arrays,
 *                          \c std::vector, \c std::map).
 */
template <class model_tmpl>
class linear_binary
{
public:
    /// The type of a model.
    typedef model_tmpl model_type;
    /// The type of a feature weight.
    typedef typename model_type::value_type value_type;

protected:
    /// The model.
    model_type& m_model;
    /// The score of this instance.
    value_type m_score;

public:
    /**
     * Constructs an object.
     *  @param  model       The model associated with the classifier.
     */
    linear_binary(model_type& model)
        : m_model(model)
    {
        clear();
    }

    /**
     * Destructs an object.
     */
    virtual ~linear_binary()
    {
    }

    /**
     * Resets the classification result.
     *  Call this function before sending a next classification instance.
     */
    inline void clear()
    {
        m_score = 0.;
    }

    /**
     * Returns the binary label of the classification result.
     *  @return bool        The binary label: \c true for positive, and
     *                      \c false for negative.
     */
    inline operator bool() const
    {
        return (0. < m_score);
    }

    /**
     * Returns the score of the classification result.
     *  @return value_type  The score, which is equivalent to the inner
     *                      product of the feature vector with the model.
     */
    inline value_type score() const
    {
        return m_score;
    }

    /**
     * Applies a scaling factor to the score.
     *  @param  scale       The scaling factor.
     */
    inline void scale(const value_type& scale)
    {
        m_score *= scale;
    }

    /**
     * Sets an attribute and value for the classification.
     *
     *  This function adds (model[a] * value) to the score.
     *  
     *  @param  a           The attribute identifier.
     *  @param  value       The attribute value.
     */
    template <class attribute_type>
    inline void set(const attribute_type& a, const value_type& value)
    {
        m_score += (m_model[a] * value);
    }

    /**
     * Computes the inner product between a feature vector and the model.
     *
     *  A feature vector is represented by a range of iterators [first, last),
     *  whose element \c *it is compatible with \c std::pair. The member
     *  \c it->first presents a feature identifier, and the member
     *  \c it->second presents the feature value.
     *  
     *  @param  first       The iterator for the first element of attributes.
     *  @param  last        The iterator for the element just beyond the
     *                      last element of attributes.
     */
    template <class iterator_type>
    inline void inner_product(iterator_type first, iterator_type last)
    {
        this->clear();
        for (iterator_type it = first;it != last;++it) {
            this->set(it->first, it->second);
        }
    }

    /**
     * Returns the name of this classifier.
     *  @return const char* The name of the classifier.
     */
    static const char *name()
    {
        const static char *str = "linear classifier (binary)";
        return str;
    }
};



/**
 * A template class for linear binary classifiers with logistic-sigmoid
 * error function.
 *
 *  This template class implements a loss function for training algorithms.
 *
 *  @param  model_tmpl      The type of a model (container of feature weights).
 *                          Any type that yields a feature weight with
 *                          operator \c [] is usable (e.g., arrays,
 *                          \c std::vector, \c std::map).
 */
template <class model_tmpl>
class linear_binary_logistic : public linear_binary<model_tmpl>
{
public:
    /// The type of a model.
    typedef model_tmpl model_type;
    /// The type of a feature weight.
    typedef typename model_type::value_type value_type;
    /// Tne type of the base class.
    typedef linear_binary<model_tmpl> base_type;

public:
    /**
     * Constructs an object.
     *  @param  model       The model associated with the classifier.
     */
    linear_binary_logistic(model_type& model)
        : base_type(model)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~linear_binary_logistic()
    {
    }

    /**
     * Computes the probability of the instance being positive.
     *  @return value_type  The probability.
     */
    inline value_type prob() const
    {
        return (
            (-100. < this->m_score) ?
            (1. / (1. + std::exp(-this->m_score))) :
            0.
            );
    }

    /**
     * Computes the error of the classification result.
     *  @param  b           The reference label for this instance.
     *  @return value_type  The error.
     */
    inline value_type error(bool b) const
    {
        value_type p = 0.;
        const value_type score = this->m_score;
        if (score < -30.) {
            p = 0.;
        } else if (30. < score) {
            p = 1.;
        } else {
            p = 1. / (1. + std::exp(-score));
        }
        return (p - static_cast<double>(b));
    }

    /**
     * Computes the error and loss of the classification result.
     *  @param  b           The reference label for this instance.
     *  @param  loss        The negative of the log of the probability of the
     *                      instance being classified to the reference label.
     *  @return value_type  The error.
     */
    inline value_type error(bool b, value_type& loss) const
    {
        value_type p = 0.;
        const value_type score = this->m_score;
        if (score < -30.) {
            p = 0.;
            loss = -static_cast<double>(b) * score;
        } else if (30. < score) {
            p = 1.;
            loss = -(static_cast<double>(b) - 1.) * score;
        } else {
            p = 1. / (1. + std::exp(-score));
            loss = b ? -std::log(p) : -std::log(1.-p);
        }
        return (p - static_cast<double>(b));
    }

    /**
     * Returns the name of this classifier.
     *  @return const char* The name of the classifier.
     */
    static const char *name()
    {
        const static char *str = "linear classifier (binary) with logistic loss";
        return str;
    }
};



/**
 * A template class for linear binary classifiers with hinge error function.
 *
 *  This template class implements a loss function for training algorithms.
 *
 *  @param  model_tmpl      The type of a model (container of feature weights).
 *                          Any type that yields a feature weight with
 *                          operator \c [] is usable (e.g., arrays,
 *                          \c std::vector, \c std::map).
 */
template <class model_tmpl>
class linear_binary_hinge : public linear_binary<model_tmpl>
{
public:
    /// The type of a model.
    typedef model_tmpl model_type;
    /// The type of a feature weight.
    typedef typename model_type::value_type value_type;
    /// Tne type of the base class.
    typedef linear_binary<model_tmpl> base_type;

public:
    /**
     * Constructs an object.
     *  @param  model       The model associated with the classifier.
     */
    linear_binary_hinge(model_type& model)
        : base_type(model)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~linear_binary_hinge()
    {
    }

    /**
     * Computes the error of the classification result.
     *  @param  b           The reference label for this instance.
     *  @return value_type  The error.
     */
    inline value_type error(bool b) const
    {
        value_type loss;
        return this->error(b, loss);
    }

    /**
     * Computes the error and loss of the classification result.
     *  @param  b           The reference label for this instance.
     *  @param  loss        The loss.
     *  @return value_type  The error.
     */
    inline value_type error(bool b, value_type& loss) const
    {
        value_type y = static_cast<value_type>(static_cast<int>(b) * 2 - 1);
        loss = 1.0 - y * this->score();
        if (0 < loss) {
            return -y;
        } else {
            loss = 0;
            return 0;
        }
    }

    /**
     * Returns the name of this classifier.
     *  @return const char* The name of the classifier.
     */
    static const char *name()
    {
        const static char *str = "linear classifier (binary) with hinge loss";
        return str;
    }
};

};

};


#endif/*__CLASSIAS_CLASSIFY_LINEAR_BINARY_H__*/
