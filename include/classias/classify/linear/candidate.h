#ifndef __CLASSIAS_CLASSIFY_LINEAR_CANDIDATE_H__
#define __CLASSIAS_CLASSIFY_LINEAR_CANDIDATE_H__

#include <cmath>

namespace classias
{

namespace classify
{

/**
 * Linear candidate classifier.
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
class linear_candidate
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
    /// The type representing an array of labels.
    typedef std::vector<label_type> labels_type;

    /// The model.
    model_type& m_model;
    /// The scores of labels.
    scores_type m_scores;
    /// The labels.
    labels_type m_labels;
    /// The index of the label that gives the highest score.
    int         m_argmax;

public:
    /**
     * Constructs an instance.
     *  @param  model       The model associated with the classifier.
     */
    linear_candidate(model_type& model)
        : m_model(model), m_argmax(-1)
    {
        clear();
    }

    /**
     * Destructs an object.
     */
    virtual ~linear_candidate()
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
            m_labels[i] = 0;
        }
    }

    /**
     * Reserves the working space for n candidates.
     *  @param  n           The number of candidates.
     */
    inline void resize(int n)
    {
        m_scores.resize(n);
        m_labels.resize(n);
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
    inline value_type score(int i) const
    {
        return m_scores[i];
    }

    /**
     * Sets the actual label of a candidate.
     *  @param  i           The candidate index.
     *  @param  l           The label.
     */
    inline void set_label(int i, const label_type& l)
    {
        m_labels[i] = l;
    }

    /**
     * Returns the actual label of a candidate.
     *  @param  i                   The candidate index.
     *  @return const label_type&   The label.
     */
    inline const label_type& label(int i) const
    {
        return m_labels[i];
    }

    /**
     * Sets an attribute for a candidate.
     *  @param  i           The candidate index.
     *  @param  a           The attribute identifier.
     *  @param  value       The attribute value.
     */
    inline void operator()(int i, const attribute_type& a, const value_type& value)
    {
        m_scores[i] += m_model[a] * value;
    }

    /**
     * Sets an array of attributes for a candidate.
     *  @param  i           The candidate index.
     *  @param  first       The iterator for the first element of attributes.
     *  @param  last        The iterator for the element just beyond the
     *                      last element of attributes.
     *  @param  reset       Specify \c true to reset the current result
     *                      before computing the inner product.
     */
    template <class iterator_type>
    inline void inner_product(
        int i, iterator_type first, iterator_type last, bool reset=true)
    {
        if (reset) {
            m_scores[i] = 0.;
        }
        for (iterator_type it = first;it != last;++it) {
            this->operator()(i, it->first, it->second);
        }
    }

    /**
     * Finalize the classification.
     */
    inline void finalize()
    {
        // Make sure that this instance is not empty.
        m_argmax = -1;
        if (m_scores.size() == 0) {
            return;
        }

        // Find the argmax index.
        m_argmax = 0;
        value_type vmax = m_scores[0];
        for (int i = 0;i < this->size();++i) {
            if (vmax < m_scores[i]) {
                m_argmax = i;
                vmax = m_scores[i];
            }
        }
    }
};

/**
 * Linear candidate classifier with logistic loss (aka, maximum entropy).
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
class linear_candidate_logistic :
    public linear_candidate<attribute_tmpl, label_tmpl, value_tmpl, model_tmpl>
{
public:
    typedef linear_candidate<attribute_tmpl, label_tmpl, value_tmpl, model_tmpl> base_type;

protected:
    /// The partition factor.
    value_type  m_norm;
    /// The probabilities of candidates.
    scores_type m_probs;

public:
    /**
     * Constructs an instance.
     *  @param  model       The model associated with the classifier.
     */
    linear_candidate_logistic(model_type& model) : base_type(model)
    {
        clear();
    }

    /**
     * Destructs an object.
     */
    virtual ~linear_candidate_logistic()
    {
    }

    /**
     * Resets the classification result.
     */
    inline void clear()
    {
        base_type::clear();

        m_norm = 0.;
        for (int i = 0;i < this->size();++i) {
            m_probs[i] = 0.;
        }
    }

    /**
     * Reserves the working space for n candidates.
     *  @param  n           The number of candidates.
     */
    inline void resize(int n)
    {
        base_type::resize(n);
        m_probs.resize(n);
    }

    /**
     * Returns the probability of a candidate.
     *  @param  i           The candidate index.
     *  @return value_type  The probability.
     */
    inline value_type prob(int i)
    {
        return m_probs[i];
    }

    template <class iterator_type>
    inline void add_to(value_type* v, iterator_type first, iterator_type last, value_type value)
    {
        for (iterator_type it = first;it != last;++it) {
            v[it->first] += value * it->second;
        }        
    }

    /**
     * Finalize the classification.
     */
    inline void finalize()
    {
        base_type::finalize();

        // Compute the exponents of scores.
        for (int i = 0;i < this->size();++i) {
            m_probs[i] = std::exp(m_scores[i]);
        }

        // Compute the partition factor, starting from the maximum value.
        m_norm = m_probs[m_argmax];
        for (int i = 0;i < this->size();++i) {
            if (i != m_argmax) {
                m_norm += m_probs[i];
            }
        }

        // Normalize the probabilities.
        for (int i = 0;i < this->size();++i) {
            m_probs[i] /= m_norm;
        }
    }
};

};

};

#endif/*__CLASSIAS_CLASSIFY_LINEAR_CANDIDATE_H__*/
