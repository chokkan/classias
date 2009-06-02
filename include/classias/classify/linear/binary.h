#ifndef __CLASSIAS_CLASSIFY_LINEAR_BINARY_H__
#define __CLASSIAS_CLASSIFY_LINEAR_BINARY_H__

#include <cmath>

namespace classias
{

namespace classify
{

/**
 * Linear binary classifier.
 *
 *  @param  attribute_tmpl  The type of an attribute.
 *  @param  value_tmpl      The type of a feature weight.
 *  @param  model_tmpl      The type of a model (array of feature weights).
 */
template <
    class attribute_tmpl,
    class value_tmpl,
    class model_tmpl
>
class linear_binary
{
public:
    /// The type of an attribute.
    typedef attribute_tmpl attribute_type;
    /// The type of a feature weight.
    typedef value_tmpl value_type;
    /// The type of a model.
    typedef model_tmpl model_type;

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
     */
    inline void clear()
    {
        m_score = 0.;
    }

    /**
     * Returns the binary label of the classification result.
     *  @return bool        The binary label.
     */
    inline operator bool() const
    {
        return (0. < m_score);
    }

    /**
     * Returns the score of the classification result.
     *  @return value_type  The score.
     */
    inline value_type score() const
    {
        return m_score;
    }

    /**
     * Sets an attribute for the classification.
     *  @param  a           The attribute identifier.
     *  @param  value       The attribute value.
     */
    inline void operator()(const attribute_type& a, const value_type& value)
    {
        m_score += m_model[a] * value;        
    }

    /**
     * Sets an array of attributes for the classification.
     *  @param  first       The iterator for the first element of attributes.
     *  @param  last        The iterator for the element just beyond the
     *                      last element of attributes.
     *  @param  reset       Specify \c true to reset the current result
     *                      before computing the inner product.
     */
    template <class iterator_type>
    inline void inner_product(iterator_type first, iterator_type last, bool reset=false)
    {
        if (reset) {
            this->clear();
        }
        for (iterator_type it = first;it != last;++it) {
            this->operator()(it->first, it->second);
        }
    }
};

/**
 * Linear binary classifier with logistic-sigmoid error function.
 *
 *  @param  attribute_tmpl  The type of an attribute.
 *  @param  value_tmpl      The type of a feature weight.
 *  @param  model_tmpl      The type of a model (array of feature weights).
 */
template <
    class attribute_tmpl,
    class value_tmpl,
    class model_tmpl
>
class linear_binary_logistic :
    public linear_binary<attribute_tmpl, value_tmpl, model_tmpl>
{
public:
    typedef linear_binary<attribute_tmpl, value_tmpl, model_tmpl> base_type;

public:
    /**
     * Constructs an object.
     *  @param  model       The model associated with the classifier.
     */
    linear_binary_logistic(model_type& model) : base_type(model)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~linear_binary_logistic()
    {
    }

    /**
     * Compute the probability for the instance being positive.
     *  @return value_type  The probability.
     */
    inline value_type prob() const
    {
        return (-100. < m_score ? (1. / (1. + std::exp(-m_score))) : 0.);
    }

    /**
     * Compute the error of the classification result.
     *  @param  b           The reference label for this instance.
     *  @return value_type  The error.
     */
    inline value_type error(bool b) const
    {
        double p = 0.;
        if (m_score < -100.) {
            p = 0.;
        } else if (100. < m_score) {
            p = 1.;
        } else {
            p = 1. / (1. + std::exp(-m_score));
        }
        return (static_cast<double>(b) - p);
    }

    /**
     * Compute the error of the classification result.
     *  @param  b           The reference label for this instance.
     *  @param  logp        The log of the probability of the instance being
     *                      classified to the reference label.
     *  @return value_type  The error.
     */
    inline value_type error(bool b, value_type& logp) const
    {
        double p = 0.;
        if (m_score < -100.) {
            p = 0.;
            logp = static_cast<double>(b) * m_score;
        } else if (100. < m_score) {
            p = 1.;
            logp = (static_cast<double>(b) - 1.) * m_score;
        } else {
            p = 1. / (1. + std::exp(-m_score));
            logp = b ? std::log(p) : std::log(1.-p);
        }
        return (static_cast<double>(b) - p);
    }
};

};

};


#endif/*__CLASSIAS_CLASSIFY_LINEAR_BINARY_H__*/
