/*
 *		Feature generators.
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

#ifndef __CLASSIAS_FEATURE_GENERATOR_H__
#define __CLASSIAS_FEATURE_GENERATOR_H__

#include "quark.h"

namespace classias
{



/**
 * Feature generator for candidate classification.
 *  In this class, a feature is identical to an attribute, and a label is
 *  ignored.
 *
 *  @param  attribute_tmpl  The type of an attribute.
 *  @param  label_tmpl      The type of a label.
 *  @param  feature_tmpl    The type of a feature.
 */
template <
    class attribute_tmpl,
    class label_tmpl,
    class feature_tmpl
>
class thru_feature_generator_base
{
public:
    /// The type of an attribute.
    typedef attribute_tmpl attribute_type;
    /// The type of a label.
    typedef label_tmpl label_type;
    /// The type of a feature.
    typedef feature_tmpl feature_type;

protected:
    /// The total number of attributes.
    size_t m_num_attributes;

public:
    /**
     * Constructs an object.
     */
    thru_feature_generator_base() :
        m_num_attributes(0)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~thru_feature_generator_base()
    {
    }

    /**
     * Returns the name of the feature generator.
     *  @return const char* The feature name.
     */
    const char* name() const
    {
        static const char *str = "thru";
        return str;
    }

    /**
     * Returns the total number of labels.
     *  @return size_t      The total number of labels.
     */
    size_t num_labels() const
    {
        return 0;
    }

    /**
     * Returns the total number of attributes.
     *  @return size_t      The total number of attributes.
     */
    size_t num_attributes() const
    {
        return m_num_attributes;
    }

    /**
     * Returns the total number of features.
     *  @return size_t      The total number of features.
     */
    size_t num_features() const
    {
        return m_num_attributes;
    }

    /**
     * Sets the total number of attributes.
     *  @param  num_attributes  The total number of attributes.
     */
    void set_num_attributes(size_t num_attributes)
    {
        m_num_attributes = num_attributes;
    }

    /**
     * Sets the total number of labels.
     *  @param  num_labels      The total number of labels.
     */
    void set_num_labels(size_t num_labels)
    {
    }

    /**
     * Returns if this class requires registration.
     *  @return bool            This class always returns \c false.
     */
    inline bool needs_registration() const
    {
        return false;
    }

    /**
     * Registers an association between an attribute and label.
     *  @param  a               The attribute.
     *  @param  l               The label.
     *  @return feature_type    The feature for the attribute and label,
     *                          which is always identical to the attribute.
     */
    inline feature_type regist(const attribute_type& a, const label_type& l)
    {
        return a;
    }

    /**
     * Returns the feature associated with a pair of an attribute and label.
     *  @param  a               The attribute.
     *  @param  l               The label.
     *  @param  f               The feature for the attribute and label,
     *                          which is always identical to the attribute.
     *  @return bool            \c true if the pair of the attribute and label
     *                          is successfully mapped to a feature, which is
     *                          always \c true.
     */
    inline bool forward(
        const attribute_type& a,
        const label_type& l,
        feature_type& f
        ) const
    {
        f = a;
        return true;
    }

    /**
     * Returns the attribute and label associated with a feature.
     *  @param  f               The feature.
     *  @param  a               The attribute associated with the feature.
     *  @param  l               The label associated with the feature.
     *  @return bool            \c if the feature is successfully associated
     *                          with the attribute and label.
     */
    inline bool backward(
        const feature_type& f,
        attribute_type& a,
        label_type& l
        ) const
    {
        a = f;
        l = 0;
        return true;
    }
};



/**
 * Feature generator for any combinations of attributes and labels.
 *
 *  @param  attribute_tmpl  The type of an attribute.
 *  @param  label_tmpl      The type of a label.
 *  @param  feature_tmpl    The type of a feature.
 */
template <
    class attribute_tmpl,
    class label_tmpl,
    class feature_tmpl
>
class dense_feature_generator_base
{
public:
    /// The type of an attribute.
    typedef attribute_tmpl attribute_type;
    /// The type of a label.
    typedef label_tmpl label_type;
    /// The type of a feature.
    typedef feature_tmpl feature_type;

protected:
    /// The total number of labels.
    size_t m_num_labels;
    /// The total number of attributes.
    size_t m_num_attributes;

public:
    /**
     * Constructs an object.
     */
    dense_feature_generator_base() :
        m_num_labels(0), m_num_attributes(0)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~dense_feature_generator_base()
    {
    }

    /**
     * Returns the name of the feature generator.
     *  @return const char* The feature name.
     */
    const char* name() const
    {
        static const char *str = "dense";
        return str;
    }

    /**
     * Returns the total number of labels.
     *  @return size_t      The total number of labels.
     */
    size_t num_labels() const
    {
        return m_num_labels;
    }

    /**
     * Returns the total number of attributes.
     *  @return size_t      The total number of attributes.
     */
    size_t num_attributes() const
    {
        return m_num_attributes;
    }

    /**
     * Returns the total number of features.
     *  @return size_t      The total number of features.
     */
    size_t num_features() const
    {
        return m_num_labels * m_num_attributes;
    }

    /**
     * Sets the total number of attributes.
     *  @param  num_attributes  The total number of attributes.
     */
    void set_num_attributes(size_t num_attributes)
    {
        m_num_attributes = num_attributes;
    }

    /**
     * Sets the total number of labels.
     *  @param  num_labels  The total number of labels.
     */
    void set_num_labels(size_t num_labels)
    {
        m_num_labels = num_labels;
    }

    /**
     * Returns if this class requires registration.
     *  @return bool    This class always returns \c false.
     */
    inline bool needs_registration() const
    {
        return false;
    }

    /**
     * Registers an association between an attribute and label.
     *  @param  a               The attribute.
     *  @param  l               The label.
     *  @return feature_type    The feature for the attribute and label,
     *                          which is always identical to the attribute.
     */
    inline feature_type regist(const attribute_type& a, const label_type& l)
    {
        feature_type f;
        forward(a, l, f);
        return f;
    }

    /**
     * Returns the feature associated with a pair of an attribute and label.
     *  @param  a               The attribute.
     *  @param  l               The label.
     *  @param  f               The feature for the attribute and label,
     *                          which is always identical to the attribute.
     *  @return bool            \c true if the pair of the attribute and label
     *                          is successfully mapped to a feature, which is
     *                          always \c true.
     */
    inline bool forward(
        const attribute_type& a,
        const label_type& l,
        feature_type& f
        ) const
    {
        f = a * m_num_labels + l;
        return true;
    }

    /**
     * Returns the attribute and label associated with a feature.
     *  @param  f               The feature.
     *  @param  a               The attribute associated with the feature.
     *  @param  l               The label associated with the feature.
     *  @return bool            \c if the feature is successfully associated
     *                          with the attribute and label.
     */
    inline bool backward(
        const feature_type& f,
        attribute_type& a,
        label_type& l
        ) const
    {
        a = f / m_num_labels;
        l = f % m_num_labels;
        return true;
    }
};



/**
 * Feature generator for combinations of attributes and labels that exist
 *  in the training data.
 *
 *  @param  attribute_tmpl  The type of an attribute.
 *  @param  label_tmpl      The type of a label.
 *  @param  feature_tmpl    The type of a feature.
 */
template <
    class attribute_tmpl,
    class label_tmpl,
    class feature_tmpl
>
class sparse_feature_generator_base
{
public:
    /// The type of an attribute.
    typedef attribute_tmpl attribute_type;
    /// The type of a label.
    typedef label_tmpl label_type;
    /// The type of a feature.
    typedef feature_tmpl feature_type;

protected:
    /// The total number of labels.
    size_t m_num_labels;
    /// The total number of attributes.
    size_t m_num_attributes;

protected:
    /// Class for associations from (attribute, label) to feature.
    typedef quark2_base<attribute_type, label_type> feature_generator_type;
    /// Associations between (attribute, label) and features.
    feature_generator_type m_features;

public:
    /**
     * Constructs an object.
     */
    sparse_feature_generator_base() :
        m_num_labels(0), m_num_attributes(0)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~sparse_feature_generator_base()
    {
    }

    /**
     * Returns the name of the feature generator.
     *  @return const char* The feature name.
     */
    const char* name() const
    {
        static const char *str = "sparse";
        return str;
    }

    /**
     * Returns the total number of labels.
     *  @return size_t      The total number of labels.
     */
    size_t num_labels() const
    {
        return m_num_labels;
    }

    /**
     * Returns the total number of features.
     *  @return size_t      The total number of features.
     */
    size_t num_features() const
    {
        return m_features.size();
    }

    /**
     * Sets the total number of attributes.
     *  @param  num_attributes  The total number of attributes.
     */
    void set_num_attributes(size_t num_attributes)
    {
        m_num_attributes = num_attributes;
    }

    /**
     * Sets the total number of labels.
     *  @param  num_labels  The total number of labels.
     */
    void set_num_labels(size_t num_labels)
    {
        m_num_labels = num_labels;
    }

    /**
     * Returns if this class requires registration.
     *  @return bool    This class always returns \c true.
     */
    inline bool needs_registration() const
    {
        return true;
    }

    /**
     * Registers an association between an attribute and label.
     *  @param  a               The attribute.
     *  @param  l               The label.
     *  @return feature_type    The feature for the attribute and label.
     */
    inline feature_type regist(const attribute_type& a, const label_type& l)
    {
        return m_features.associate(a, l);
    }

    /**
     * Returns the feature associated with a pair of an attribute and label.
     *  @param  a               The attribute.
     *  @param  l               The label.
     *  @param  f               The feature for the attribute and label.
     *  @return bool            \c true if the pair of the attribute and label
     *                          is successfully mapped to a feature, which is
     *                          always \c true.
     */
    inline bool forward(
        const attribute_type& a,
        const label_type& l,
        feature_type& f
        ) const
    {
        f = m_features.to_value(a, l, -1);
        return (f != -1);
    }

    /**
     * Returns the attribute and label associated with a feature.
     *  @param  f               The feature.
     *  @param  a               The attribute associated with the feature.
     *  @param  l               The label associated with the feature.
     *  @return bool            \c if the feature is successfully associated
     *                          with the attribute and label.
     */
    inline bool backward(
        const feature_type& f,
        attribute_type& a,
        label_type& l
        ) const
    {
        m_features.to_item(f, a, l);
        return true;
    }
};



};

#endif/*__CLASSIAS_FEATURE_GENERATOR_H__*/
