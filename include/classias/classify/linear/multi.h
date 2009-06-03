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
 *  @param  features_tmpl   The type of a feature generator.
 */
template <
    class attribute_tmpl,
    class label_tmpl,
    class value_tmpl,
    class model_tmpl,
    class features_tmpl
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
    /// The type of a feature generator.
    typedef features_tmpl features_type;

protected:
    /// The type representing an array of scores.
    typedef std::vector<value_type> scores_type;

    /// The model.
    model_type&     m_model;
    /// The scores of labels.
    scores_type     m_scores;
    /// The feature generator.
    features_type&  m_feature_generator;
    /// The index of the label that gives the highest score.
    int             m_argmax;

public:
    /**
     * Constructs an instance.
     *  @param  model       The model associated with the classifier.
     *  @param  feature_generator   The feature generator.
     */
    linear_multi(model_type& model, features_type& feature_generator)
        : m_model(model), m_feature_generator(feature_generator)
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
        for (int i = 0;i < this->size();++i) {
            m_scores[i] = 0.;
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
     *  @return int         The index of the candidate that yields the
     *                      highest score.
     */
    inline int argmax() const
    {
        return m_argmax;
    }

    /**
     * Returns the score of a candidate.
     *  @param  i           The candidate index.
     *  @return value_type  The score.
     */
    inline value_type score(int i)
    {
        return m_scores[i];
    }

    /**
     * Sets an attribute for a candidate.
     *  @param  i           The candidate index.
     *  @param  a           The attribute identifier.
     *  @param  l           The label for the candidate.
     *  @param  value       The attribute value.
     */
    inline void operator()(int i, const attribute_type& a, const label_type& l, const value_type& value)
    {
        int_t fid = m_feature_generator.forward(a, l);
        if (0 <= fid) {
            m_scores[i] += m_model[fid] * value;
        }
    }

    /**
     * Sets an array of attributes for a candidate.
     *  @param  i           The candidate index.
     *  @param  first       The iterator for the first element of attributes.
     *  @param  last        The iterator for the element just beyond the
     *                      last element of attributes.
     *  @param  l           The label for the candidate.
     *  @param  reset       Specify \c true to reset the current result
     *                      before computing the inner product.
     */
    template <class iterator_type>
    inline void inner_product(int i, iterator_type first, iterator_type last, const label_type& l, bool reset=true)
    {
        if (reset) {
            m_scores[i] = 0.;
        }
        for (iterator_type it = first;it != last;++it) {
            this->operator()(i, it->first, l, it->second);
        }
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
        for (int i = 0;i < this->size();++i) {
            if (vmax < m_scores[i]) {
                m_argmax = i;
                vmax = m_scores[i];
            }
        }
    }
};

/**
 * Linear multi-class classifier with sigmoid function (maximum entropy).
 *
 *  @param  attribute_tmpl  The type of an attribute.
 *  @param  label_tmpl      The type of a label.
 *  @param  value_tmpl      The type of a feature weight.
 *  @param  model_tmpl      The type of a model (array of feature weights).
 *  @param  features_tmpl   The type of a feature generator.
 */
template <
    class attribute_tmpl,
    class label_tmpl,
    class value_tmpl,
    class model_tmpl,
    class features_tmpl
>
class linear_multi_logistic :
    public linear_multi<attribute_tmpl, label_tmpl, value_tmpl, model_tmpl, features_tmpl>
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
    /// The type of a feature generator.
    typedef features_tmpl features_type;
    /// The type of the base class.
    typedef linear_multi<attribute_tmpl, label_tmpl, value_tmpl, model_tmpl, features_tmpl> base_type;

protected:
    value_type  m_lognorm;

public:
    /**
     * Constructs an instance.
     *  @param  model       The model associated with the classifier.
     *  @param  feature_generator   The feature generator.
     */
    linear_multi_logistic(model_type& model, features_tmpl& feature_generator)
        : base_type(model, feature_generator), m_lognorm(0)
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
     *  @param  i           The candidate index.
     *  @return value_type  The probability.
     */
    inline value_type prob(int i)
    {
        return std::exp(m_scores[i] - m_lognorm);
    }

    /**
     * Returns the log of the probability for a label.
     *  @param  i           The candidate index.
     *  @return value_type  The probability.
     */
    inline value_type logprob(int i)
    {
        return (m_scores[i] - m_lognorm);
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
        value_type max = m_scores[this->m_argmax];
        for (int i = 0;i < this->size();++i) {
            sum += std::exp(m_scores[i] - max);
        }
        m_lognorm = max + std::log(sum);
    }
};

};

};

#endif/*__CLASSIAS_CLASSIFY_LINEAR_MULTI_H__*/
