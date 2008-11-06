#ifndef __LINEAR_H__
#define __LINEAR_H__

#include <vector>

namespace classias
{

/**
 * Score base class.
 *  @param  value_base          The type representing a score and probability.
 */
template <class value_base>
class score_base
{
public:
    typedef value_base value_type;

protected:
    value_type m_score;
    value_type m_probability;

public:
    score_base()
    {
        clear();
    }

    virtual ~score_base()
    {
    }

    inline void clear()
    {
        m_score = 0;
        m_probability = 0;
    }

    inline value_type& get_score()
    {
        return m_score;
    }

    inline void set_score(const value_type& value)
    {
        m_score = value;
    }

    inline void add_score(const value_type& value)
    {
        m_score += value;
    }

    inline value_type& get_probability()
    {
        return m_probability;
    }

    inline void set_probability(const value_type& value)
    {
        m_probability = value;
    }
};



template <class feature_model_base>
class linear_score_feature_base
{
public:
    /// 
    typename feature_model_base model_type;
    /// A type representing an element name.
    typedef typename model_type::identifier_type identifier_type;
    /// A type representing an element value.
    typedef typename model_type::value_type value_type;
    ///
    typedef score_base<value_type> score_type;

protected:
    ///
    const model_type& m_model;
    /// The score of the linear combination.
    score_type m_score;

public:
    /**
     * Constructs a linear scorer.
     */
    linear_score_feature_base(const model_type& model)
        : m_model(model)
    {
        clear();
    }

    /**
     * Destructs the linear scorer
     */
    virtual ~linear_score_feature_base()
    {
    }

    /**
     * Initialize the scorer.
     */
    inline void clear()
    {
        m_score.clear();
    }

    /**
     * Adds an element (name, value) to the end of the vector.
     *  @param  name        The element name.
     *  @param  value       The element value.
     */
    inline void append(const identifier_type& name, const value_type& value)
    {
        m_score.add_score(m_model[name] * value);
    }
};



template <class attribute_model_base>
class linear_score_attribute_base
{
public:
    /// 
    typename attribute_model_base model_type;
    /// A type representing an element name.
    typedef typename model_type::identifier_type identifier_type;
    /// A type representing an instance label.
    typedef typename model_type::label_type label_type;
    /// A type representing an element value.
    typedef typename model_type::value_type value_type;
    ///
    typedef score_base<value_type> score_type;
    ///
    typedef std::vector<score_type> scores_type;

protected:
    ///
    const model_type& m_model;
    /// The score of the linear combination.
    value_type m_score;

public:
    /**
     * Constructs a linear scorer.
     */
    linear_feature_score_base(const model_type& model)
        : m_model(model)
    {
        clear();
    }

    /**
     * Destructs the linear scorer
     */
    virtual ~linear_feature_score_base()
    {
    }

    /**
     * Initialize the scorer.
     */
    inline void clear()
    {
        score = 0;
    }

    /**
     * Adds an element (name, value) to the end of the vector.
     *  @param  name        The element name.
     *  @param  value       The element value.
     */
    inline void append(const identifier_type& name, const value_type& value)
    {
        score += m_model[name] * value;
    }
};

};

#endif/*__LINEAR_H__*/