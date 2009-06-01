/*
 *		Instance collections.
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
 *  @param  instance_tmpl           The type of an instance.
 *  @param  attributes_quark_tmpl   The type of an attribute quark.
 */
template <
    class instance_tmpl,
    class attributes_quark_tmpl
>
class binary_data_base
{
public:
    /// The type of an instance.
    typedef instance_tmpl instance_type;
    /// The type of a feature vector.
    typedef attributes_quark_tmpl attributes_quark_type;
    /// The type of an attribute.
    typedef typename attributes_quark_type::value_type attribute_type;

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
    /// A feature quark.
    attributes_quark_type attributes;
    /// The start index of features.
    int_t feature_start_index;

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
     * Returns a read/write reference to an instance.
     *  @param  i               The index number for an instance.
     *  @retval instance_type&  Reference to the instance.
     */
    inline instance_type& operator[](size_type i)
    {
        return instances[i];
    }

    /**
     * Returns a read-only reference to an instance.
     *  @param  i                       The index number for an instance.
     *  @retval const instance_type&    Reference to the instance.
     */
    inline const instance_type& operator[](size_type i) const
    {
        return instances[i];
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
        instances.push_back(instance_type());
        return this->back();
    }

    /**
     * Sets the start index of user features.
     *  @param  index       The start index of user features.
     */
    inline void set_user_feature_start(int_t index)
    {
        feature_start_index = index;
    }

    /**
     * Returns the start index of user features.
     *  @return fid_type    The start index of user features.
     */
    inline int_t get_user_feature_start() const
    {
        return feature_start_index;
    }

    /**
     * Returns the total number of attributes.
     *  @return int         The total number of attributes.
     */
    int_t num_attributes() const
    {
        return attributes.size();
    }

    /**
     * Returns the total number of features.
     *  @return int         The total number of features.
     */
    int_t num_features() const
    {
        return attributes.size();
    }

    /**
     * Returns the total number of labels.
     *  @return int         The total number of labels.
     */
    int_t num_labels() const
    {
        return 2;
    }

    /**
     * Finalize the data set.
     */
    void finalize()
    {
    }
};



/**
 * Data set for candidate instances.
 *
 *  This class provides a data set for candidate instances.
 *
 *  @param  instance_tmpl       The type of an instance.
 *  @param  features_quark_tmpl The type of a feature quark.
 *  @param  label_quark_tmpl    The type of a label quark.
 */
template <
    class instance_tmpl,
    class features_quark_tmpl,
    class labels_quark_tmpl
>
class candidate_data_base :
    public binary_data_base<instance_tmpl, features_quark_tmpl>
{
public:
    /// The type of label quark.
    typedef labels_quark_tmpl labels_quark_type;
    /// The type of a label.
    typedef typename labels_quark_type::value_type label_type;
    /// The type of a container for positive labels.
    typedef std::vector<label_type> positive_labels_type;

    /// A set of labels in the data set.
    labels_quark_type labels;
    /// A set of positive labels in the data set.
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

    /**
     * Appends a positive label.
     *  @param  l       The positive label to append.
     */
    void append_positive_label(label_type l)
    {
        positive_labels.push_back(l);
    }

    /**
     * Returns the total number of labels.
     *  @return int         The total number of labels.
     */
    int_t num_labels() const
    {
        return labels.size();
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
 *  @param  feature_generator_tmpl  The type of a feature generator.
 */
template <
    class instance_tmpl,
    class attributes_quark_tmpl,
    class label_quark_tmpl,
    class feature_generator_tmpl
>
class multi_data_base :
    public candidate_data_base<instance_tmpl, attributes_quark_tmpl, label_quark_tmpl>
{
public:
    /// The type of the feature-generator class.
    typedef feature_generator_tmpl feature_generator_type;
    /// The feature generator.
    feature_generator_type feature_generator;

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
     * Returns the total number of features.
     *  @return int         The total number of features.
     */
    int_t num_features() const
    {
        return feature_generator.num_features();
    }

    /**
     * Finalize the data set.
     */
    void finalize()
    {
        feature_generator.set_num_labels(this->labels.size());
        feature_generator.set_num_attributes(this->attributes.size());

        if (feature_generator.needs_registration()) {
            typename iterator iti;
            for (iti = this->begin();iti != this->end();++iti) {
                typename instance_type::iterator it;
                for (it = iti->begin();it != iti->end();++it) {
                    this->feature_generator.regist(it->first, iti->get_label());
                }
            }
        }
    }
};

};

#endif/*__CLASSIAS_DATA_H__*/
