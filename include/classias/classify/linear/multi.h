/*
 *		Multi-class classifier with linear models.
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

#ifndef __CLASSIAS_CLASSIFY_LINEAR_MULTI_H__
#define __CLASSIAS_CLASSIFY_LINEAR_MULTI_H__

#include <cmath>

namespace classias
{

namespace classify
{

/**
 * Linear multi-class classifier.
 *
 *  @param  attribute_tmpl  The type of an attribute.
 *  @param  label_tmpl      The type of a label.
 *  @param  value_tmpl      The type of a feature weight.
 *  @param  model_tmpl      The type of a model (array of feature weights).
 */
template <
    class attribute_tmpl,
    class label_tmpl,
    class value_tmpl,
    class model_tmpl
>
class linear_multi
{
public:
    /// The type of an attribute.
    typedef attribute_tmpl attribute_type;
    /// The type of a label.
    typedef label_tmpl label_type;
    /// The type of a feature weight.
    typedef value_tmpl value_type;
    /// The type of a model.
    typedef model_tmpl model_type;

protected:
    /// The type representing an array of scores.
    typedef std::vector<value_type> scores_type;

    /// The model.
    model_type&     m_model;
    /// The scores of labels.
    scores_type     m_scores;
    /// The index of the label that gives the highest score.
    int             m_argmax;

public:
    /**
     * Constructs an instance.
     *  @param  model       The model associated with the classifier.
     */
    linear_multi(model_type& model)
        : m_model(model)
    {
        clear();
    }

    /**
     * Destructs an object.
     */
    virtual ~linear_multi()
    {
    }

    /**
     * Resets the classification result.
     */
    inline void clear()
    {
        m_argmax = -1;
        for (int l = 0;l < this->size();++l) {
            m_scores[l] = 0.;
        }
    }

    /**
     * Reserves the working space for n candidates.
     *  @param  n           The number of candidates.
     */
    inline void resize(int n)
    {
        m_scores.resize(n);
    }

    /**
     * Returns the number of candidates.
     *  @return int         The number of candidates.
     */
    inline int size() const
    {
        return (int)m_scores.size();
    }

    /**
     * Returns the argmax index.
     *  @return int         The label of the candidate that yields the
     *                      highest score.
     */
    inline int argmax() const
    {
        return m_argmax;
    }

    /**
     * Returns the score of a candidate.
     *  @param  l           The label for the candidate.
     *  @return value_type  The score.
     */
    inline value_type score(int l)
    {
        return m_scores[l];
    }

    /**
     * Sets an attribute for a candidate.
     *  @param  fgen        The feature generator.
     *  @param  a           The attribute identifier.
     *  @param  l           The label for the candidate.
     *  @param  value       The attribute value.
     */
    template <class feature_generator_type>
    inline void set(
        feature_generator_type& fgen,
        const attribute_type& a,
        int l,
        const value_type& value
        )
    {
        typename feature_generator_type::feature_type f = fgen.forward(a, l);
        if (0 <= f) {
            m_scores[l] += m_model[f] * value;
        }
    }

    /**
     * Sets an array of attributes for a candidate.
     *  @param  l           The label for the candidate.
     *  @param  first       The iterator for the first element of attributes.
     *  @param  last        The iterator for the element just beyond the
     *                      last element of attributes.
     *  @param  reset       Specify \c true to reset the current result
     *                      before computing the inner product.
     */
    template <class feature_generator_type, class iterator_type>
    inline void inner_product(int l, feature_generator_type& fgen, iterator_type first, iterator_type last)
    {
        m_scores[l] = 0.;
        for (iterator_type it = first;it != last;++it) {
            this->set(fgen, it->first, l, it->second);
        }
    }

    template <class feature_generator_type, class iterator_type>
    inline void inner_product_scaled(int l, feature_generator_type& fgen, iterator_type first, iterator_type last, const value_type& scale)
    {
        this->inner_product(l, fgen, first, last);
        this->scale(l, scale);
    }

    inline void scale(int l, const value_type& scale)
    {
        m_scores[l] *= scale;
    }

    /**
     * Finalize the classification.
     */
    inline void finalize()
    {
        if (this->size() == 0) {
            return;
        }

        // Find the argmax index.
        m_argmax = 0;
        double vmax = m_scores[0];
        for (int l = 0;l < this->size();++l) {
            if (vmax < m_scores[l]) {
                m_argmax = l;
                vmax = m_scores[l];
            }
        }
    }

    static const char *name()
    {
        const static char *str = "linear classifier (multi)";
        return str;
    }
};

/**
 * Linear multi-class classifier with sigmoid function (maximum entropy).
 *
 *  @param  attribute_tmpl  The type of an attribute.
 *  @param  value_tmpl      The type of a feature weight.
 *  @param  model_tmpl      The type of a model (array of feature weights).
 */
template <
    class attribute_tmpl,
    class label_tmpl,
    class value_tmpl,
    class model_tmpl
>
class linear_multi_logistic :
    public linear_multi<attribute_tmpl, label_tmpl, value_tmpl, model_tmpl>
{
public:
    /// The type of an attribute.
    typedef attribute_tmpl attribute_type;
    /// The type of a label.
    typedef label_tmpl label_type;
    /// The type of a feature weight.
    typedef value_tmpl value_type;
    /// The type of a model.
    typedef model_tmpl model_type;
    /// The type of the base class.
    typedef linear_multi<attribute_tmpl, label_tmpl, value_tmpl, model_tmpl> base_type;

protected:
    value_type  m_lognorm;

public:
    /**
     * Constructs an instance.
     *  @param  model       The model associated with the classifier.
     *  @param  feature_generator   The feature generator.
     */
    linear_multi_logistic(model_type& model)
        : base_type(model), m_lognorm(0)
    {
        clear();
    }

    /**
     * Destructs an instance.
     */
    virtual ~linear_multi_logistic()
    {
    }

    /**
     * Resets the classification result.
     */
    inline void clear()
    {
        base_type::clear();
        m_lognorm = 0.;
    }

    /**
     * Returns the probability for a label.
     *  @param  l           The label for the candidate.
     *  @return value_type  The probability.
     */
    inline value_type prob(int l)
    {
        return std::exp(this->m_scores[l] - m_lognorm);
    }

    /**
     * Returns the log of the probability for a label.
     *  @param  l           The label for the candidate.
     *  @return value_type  The probability.
     */
    inline value_type logprob(int l)
    {
        return (this->m_scores[l] - m_lognorm);
    }

    inline value_type error(int l, int r)
    {
        return prob(l) - (l == r);
    }

    /**
     * Finalize the classification.
     */
    inline void finalize()
    {
        base_type::finalize();

        if (this->size() == 0) {
            return;
        }

        // Compute the partition factor, starting from the maximum value.
        value_type sum = 0.;
        value_type max = this->m_scores[this->m_argmax];
        for (int l = 0;l < this->size();++l) {
            sum += std::exp(this->m_scores[l] - max);
        }
        m_lognorm = max + std::log(sum);
    }

    static const char *name()
    {
        const static char *str = "linear classifier (multi) with logistic loss";
        return str;
    }
};

};

};

#endif/*__CLASSIAS_CLASSIFY_LINEAR_MULTI_H__*/
