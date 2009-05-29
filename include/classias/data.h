/*
 *		Instance collections for Classias.
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
 *     * Neither the name of the Northwestern University, University of Tokyo,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written
 *       permission.
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

#ifndef __CLASSIAS_DATA_H__
#define __CLASSIAS_DATA_H__

#include <vector>

namespace classias
{

/**
 * Collection class of binary-classification instances.
 *
 *  This class provides a data set for binary classification.
 *
 *  @param  instance_tmpl       The type of an instance.
 *  @param  features_quark_tmpl The type of a feature quark.
 */
template <
    class instance_tmpl,
    class features_quark_tmpl
>
class binary_data_base
{
public:
    /// The type of an instance.
    typedef instance_tmpl instance_type;
    /// The type of the traits class associated with the instance class.
    typedef typename instance_type::traits_type traits_type;

    /// The type of a feature vector.
    typedef features_quark_tmpl features_quark_type;
    /// The type of a feature.
    typedef typename features_quark_type::value_type feature_type;

    /// A type providing a container of instances.
    typedef std::vector<instance_type> instances_type;
    /// A type counting the number of pairs in a container.
    typedef typename instances_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename instances_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename instances_type::const_iterator const_iterator;

    /// A container of instances.
    instances_type instances;
    /// A data traits.
    traits_type traits;
    /// A feature quark.
    features_quark_type features;
    /// The start index of features.
    feature_type feature_start_index;

    /**
     * Constructs the object.
     */
    binary_data_base() : feature_start_index(0)
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~binary_data_base()
    {
    }

    /**
     * Erases all the instances of the data.
     */
    inline void clear()
    {
        instances.clear();
    }

    /**
     * Tests if the data is empty.
     *  @retval bool        \c true if the data is empty,
     *                      \c false otherwise.
     */
    inline bool empty() const
    {
        return instances.empty();
    }

    /**
     * Returns the number of instances in the data.
     *  @retval size_type   The current size of the data.
     */
    inline size_type size() const
    {
        return instances.size();
    }

    /**
     * Returns a random-access iterator to the first instance.
     *  @retval iterator    A random-access iterator (for read/write)
     *                      addressing the first instance in the data or
     *                      to the location succeeding an empty instance.
     */
    inline iterator begin()
    {
        return instances.begin();
    }

    /**
     * Returns a random-access iterator to the first instance.
     *  @retval iterator    A random-access iterator (for read-only)
     *                      addressing the first instance in the data or
     *                      to the location succeeding an empty instance.
     */
    inline const_iterator begin() const
    {
        return instances.begin();
    }

    /**
     * Returns a random-access iterator pointing just beyond the last instance.
     *  @retval iterator    A random-access iterator (for read/write)
     *                      addressing the end of the instance.
     */
    inline iterator end()
    {
        return instances.end();
    }

    /**
     * Returns a random-access iterator pointing just beyond the last instance.
     *  @retval iterator    A random-access iterator (for read-only)
     *                      addressing the end of the instance.
     */
    inline const_iterator end() const
    {
        return instances.end();
    }

    /**
     * Returns the reference to the last instance.
     *  @retval instance_type&  The reference pointing to the last instance
     *                          in the data.
     */
    inline instance_type& back()
    {
        return instances.back();
    }

    /**
     * Creates and returns a new instance.
     *  @retval instance_type&  The reference to the new instance.
     */
    inline instance_type& new_element()
    {
        instances.push_back(instance_type(&traits));
        return this->back();
    }

    inline void set_user_feature_start(feature_type index)
    {
        feature_start_index = index;
    }

    inline feature_type get_user_feature_start() const
    {
        return feature_start_index;
    }

    /**
     * Updates the information in the traits class.
     */
    void finalize()
    {
        // The number of labels is 2 (binary).
        traits.set_num_labels(2);
        // Features and attributes are equivalent.
        traits.set_num_attributes(this->features.size());
    }
};





/**
 * Data set for ranking instances.
 *
 *  This class provides a data set for ranking instances.
 *
 *  @param  instance_tmpl       The type of an instance.
 *  @param  features_quark_tmpl The type of a feature quark.
 *  @param  label_quark_tmpl    The type of a label quark.
 */
template <
    class instance_tmpl,
    class features_quark_tmpl,
    class label_quark_tmpl
>
class candidate_data_base : public binary_data_base<instance_tmpl, features_quark_tmpl>
{
public:
    /// The type of the base class.
    typedef binary_data_base<instance_tmpl, features_quark_tmpl> base_type;
    /// The type of label quark.
    typedef label_quark_tmpl label_quark_type;
    /// The type of a label.
    typedef typename label_quark_type::value_type label_type;
    /// The 
    typedef typename base_type::size_type size_type;

    typedef std::vector<label_type> positive_labels_type;

    /// A set of labels.
    label_quark_type labels;
    /// 
    positive_labels_type positive_labels;

    /**
     * Constructs the object.
     */
    candidate_data_base()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~candidate_data_base()
    {
    }

    void append_positive_label(label_type l)
    {
        positive_labels.push_back(l);
    }

    /**
     * Updates the information in the traits class.
     */
    void finalize()
    {
        // The number of distinct labels.
        this->traits.set_num_labels(this->labels.size());
        // Features and attributes are equivalent.
        this->traits.set_num_attributes(this->features.size());
    }
};


/**
 * Data set for classification instances.
 *
 *  This class provides a data set for classification (attribute-label)
 *  instances.
 *
 *  @param  instance_tmpl           The type of an instance.
 *  @param  attributes_quark_tmpl   The type of an attribute quark.
 *  @param  label_quark_tmpl        The type of a label quark.
 */
template <
    class instance_tmpl,
    class attributes_quark_tmpl,
    class label_quark_tmpl
>
class multi_data_base :
    public candidate_data_base<instance_tmpl, attributes_quark_tmpl, label_quark_tmpl>
{
public:
    /// The type of the base class.
    typedef candidate_data_base<instance_tmpl, attributes_quark_tmpl, label_quark_tmpl> base_type;
    /// The type of the instance class.
    typedef typename base_type::instance_type instance_type;

public:
    /**
     * Constructs the object.
     */
    multi_data_base()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~multi_data_base()
    {
    }

    /**
     * Updates the information in the traits class.
     */
    void finalize()
    {
        // The number of distinct labels.
        this->traits.set_num_labels(this->labels.size());
        // Features are actually attributes.
        this->traits.set_num_attributes(this->features.size());

        // Check if the traits class needs to examine the instances.
        if (this->traits.needs_examination()) {
            // This actually generates (sparse) features.
            typename base_type::iterator it;
            for (it = this->begin();it != this->end();++it) {
                it->examine();
            }
        }
    }
};

};

#endif/*__CLASSIAS_DATA_H__*/
