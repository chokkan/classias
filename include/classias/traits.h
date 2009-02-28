#ifndef __CLASSIAS_TRAITS_H__
#define __CLASSIAS_TRAITS_H__

#include "quark.h"

namespace classias
{

template <
    class attribute_tmpl,
    class label_tmpl,
    class feature_tmpl
>
class feature_data_traits_base
{
public:
    typedef attribute_tmpl attribute_type;
    typedef label_tmpl label_type;
    typedef feature_tmpl feature_type;

public:
    int m_num_labels;
    int m_num_attributes;

public:
    feature_data_traits_base() :
        m_num_labels(0), m_num_attributes(0)
    {
    }

    virtual ~feature_data_traits_base()
    {
    }

    int num_labels() const
    {
        return m_num_labels;
    }

    int num_attributes() const
    {
        return m_num_attributes;
    }

    int num_features() const
    {
        return m_num_attributes;
    }

    void set_num_labels(int num_labels)
    {
        m_num_labels = num_labels;
    }

    void set_num_attributes(int num_attributes)
    {
        m_num_attributes = num_attributes;
    }

    inline bool needs_examination()
    {
        return false;
    }

    inline void examine(const attribute_type& a, const label_type& l)
    {
    }

    inline feature_type forward(const attribute_type& a, const label_type& l)
    {
        return a;
    }

    inline bool backward(
        const feature_type& f,
        attribute_type& a,
        label_type& l
        )
    {
        return false;
    }
};



template <
    class attribute_tmpl,
    class label_tmpl,
    class feature_tmpl
>
class dense_data_traits_base
{
public:
    typedef attribute_tmpl attribute_type;
    typedef label_tmpl label_type;
    typedef feature_tmpl feature_type;

public:
    int m_num_labels;
    int m_num_attributes;

public:
    dense_data_traits_base() :
        m_num_labels(0), m_num_attributes(0)
    {
    }

    virtual ~dense_data_traits_base()
    {
    }

    int num_labels() const
    {
        return m_num_labels;
    }

    int num_attributes() const
    {
        return m_num_attributes;
    }

    int num_features() const
    {
        return m_num_labels * m_num_attributes;
    }

    void set_num_labels(int num_labels)
    {
        m_num_labels = num_labels;
    }

    void set_num_attributes(int num_attributes)
    {
        m_num_attributes = num_attributes;
    }

    inline bool needs_examination()
    {
        return false;
    }

    inline void examine(const attribute_type& a, const label_type& l)
    {
    }

    inline feature_type forward(const attribute_type& a, const label_type& l)
    {
        return l * m_num_attributes + a;
    }

    inline bool backward(
        const feature_type& f,
        attribute_type& a,
        label_type& l
        )
    {
        l = f / m_num_attributes;
        a = f % m_num_attributes;
        return true;
    }
};



template <
    class attribute_tmpl,
    class label_tmpl,
    class feature_tmpl
>
class sparse_data_traits_base
{
public:
    typedef attribute_tmpl attribute_type;
    typedef label_tmpl label_type;
    typedef feature_tmpl feature_type;

    typedef quark2_base<attribute_type, label_type> feature_generator_type;

protected:
    int m_num_labels;
    int m_num_attributes;
    feature_generator_type m_features;

public:
    sparse_data_traits_base() :
        m_num_labels(0), m_num_attributes(0)
    {
    }

    virtual ~sparse_data_traits_base()
    {
    }

    int num_labels() const
    {
        return m_num_labels;
    }

    int num_attributes() const
    {
        return m_num_attributes;
    }

    int num_features() const
    {
        return m_features.size();
    }

    void set_num_labels(int num_labels)
    {
        m_num_labels = num_labels;
    }

    void set_num_attributes(int num_attributes)
    {
        m_num_attributes = num_attributes;
    }

    inline bool needs_examination()
    {
        return true;
    }

    inline void examine(const attribute_type& a, const label_type& l)
    {
        m_features.associate(a, l);
    }

    inline feature_type forward(const attribute_type& a, const label_type& l)
    {
        return m_features.to_value(a, l, -1);
    }

    inline bool backward(
        const feature_type& f,
        attribute_type& a,
        label_type& l
        )
    {
        m_features.to_item(f, a, l);
        return true;
    }
};

};

#endif/*__CLASSIAS_TRAITS_H__*/
