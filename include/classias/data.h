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

#ifndef __CLASSIAS_DATA_H__
#define __CLASSIAS_DATA_H__

#include <vector>

namespace classias
{


/**
 * A template class for a collection of binary-classification instances.
 *
 *  This class represents a data set for training a binary classifier. The
 *  class stores instances into a vector. This class implements the necessary
 *  functions num_attributes(), num_features(), and num_labels() for training
 *  algorithms. Do not forget to call set_num_features() to specify the total
 *  number of features.
 *
 *  @param  instance_tmpl           The type of an instance.
 */
template <
    class instance_tmpl
>
class binary_data_base
{
public:
    /// The type of an instance.
    typedef instance_tmpl instance_type;

    /// A type providing a container of instances.
    typedef std::vector<instance_type> instances_type;
    /// A type counting the number of pairs in a container.
    typedef typename instances_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename instances_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename instances_type::const_iterator const_iterator;
    /// The type of an attribute.
    typedef typename instance_type::attribute_type attribute_type;

protected:
    /// A container of instances.
    instances_type instances;
    /// The number of features.
    int m_num_features;
    /// The start index of features.
    int m_feature_start_index;

public:
    /**
     * Constructs the object.
     */
    binary_data_base() : m_num_features(0), m_feature_start_index(0)
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
    inline void set_user_feature_start(int index)
    {
        m_feature_start_index = index;
    }

    /**
     * Returns the start index of user features.
     *  @return fid_type    The start index of user features.
     */
    inline int get_user_feature_start() const
    {
        return m_feature_start_index;
    }

    /**
     * Sets the total number of features.
     *  @param  num         The number of features.
     */
    inline void set_num_features(int num)
    {
        m_num_features = num;
    }

    /**
     * Returns the total number of attributes.
     *  @return int         The total number of attributes.
     */
    int num_attributes() const
    {
        return m_num_features;
    }

    /**
     * Returns the total number of labels.
     *  @return int         The total number of labels. This is always 2 for
     *                      the data collection for binary instances.
     */
    int num_labels() const
    {
        return 2;
    }

    /**
     * Returns the total number of features.
     *  @return int         The total number of features.
     */
    int num_features() const
    {
        return m_num_features;
    }
};



/**
 * A template class for a collection of binary-classification instances
 * with a quark assigning attribute identifiers.
 *
 *  This class represents a data set for training a binary classifier. The
 *  class stores instances into a vector. This class implements the necessary
 *  functions num_attributes(), num_features(), and num_labels() for training
 *  algorithms. In addition, this class owns a quark to assign attribute
 *  identifiers, and overwrites num_attributes() and num_features().
 *
 *  @param  instance_tmpl           The type of an instance.
 *  @param  attributes_quark_tmpl   The type of an attribute quark.
 */
template <
    class instance_tmpl,
    class attributes_quark_tmpl
>
class binary_data_with_quark_base : public binary_data_base<instance_tmpl>
{
public:
    /// The type of an instance.
    typedef instance_tmpl instance_type;
    /// The type of a feature vector.
    typedef attributes_quark_tmpl attributes_quark_type;
    /// The type of the base class.
    typedef binary_data_base<instance_tmpl> base_type;

    /// A type counting the number of pairs in a container.
    typedef typename base_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename base_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename base_type::const_iterator const_iterator;
    /// The type of an attribute.
    typedef typename base_type::attribute_type attribute_type;

    /// A feature quark.
    attributes_quark_type attributes;

    /**
     * Constructs the object.
     */
    binary_data_with_quark_base()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~binary_data_with_quark_base()
    {
    }

    /**
     * Returns the total number of attributes.
     *  @return int         The total number of attributes.
     */
    int num_attributes() const
    {
        return attributes.size();
    }

    /**
     * Returns the total number of labels.
     *  @return int         The total number of labels. This is always 2 for
     *                      the data collection for binary instances.
     */
    int num_labels() const
    {
        return 2;
    }

    /**
     * Returns the total number of features.
     *  @return int         The total number of features.
     */
    int num_features() const
    {
        return attributes.size();
    }
};



/**
 * A template class for a collection of candidate-classification instances.
 *
 *  This class represents a data set for training a candidate classifier. The
 *  class stores instances into a vector. This class implements the necessary
 *  functions num_attributes(), num_features(), and num_labels() for training
 *  algorithms. Do not forget to call set_num_features() to specify the total
 *  number of features.
 *
 *  @param  instance_tmpl           The type of an instance.
 *  @param  features_quark_tmpl     The type of a feature quark.
 */
template <
    class instance_tmpl,
    class feature_generator_tmpl
>
class candidate_data_base :
    public binary_data_base<instance_tmpl>
{
public:
    /// The type of an instance.
    typedef instance_tmpl instance_type;
    /// The type of the feature-generator class.
    typedef feature_generator_tmpl feature_generator_type;
    /// The base class.
    typedef binary_data_base<instance_tmpl> base_type;

    /// A type providing a container of instances.
    typedef typename base_type::instances_type instances_type;
    /// A type counting the number of pairs in a container.
    typedef typename base_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename base_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename base_type::const_iterator const_iterator;
    /// The type of an attribute.
    typedef typename base_type::attribute_type attribute_type;

    /// The type of a container for positive labels.
    typedef std::vector<int> positive_labels_type;
    /// A set of positive labels in the data set.
    positive_labels_type positive_labels;
    /// The feature generator.
    feature_generator_type feature_generator;

public:
    /**
     * Constructs the object.
     */
    candidate_data_base()
    {
    }

    /**
     * Appends a positive label.
     *  @param  l       The positive label to append.
     */
    void append_positive_label(int l)
    {
        positive_labels.push_back(l);
    }

    /**
     * Destructs the object.
     */
    virtual ~candidate_data_base()
    {
    }

    /**
     * Returns the total number of attributes.
     *  @return int         The total number of attributes.
     */
    int num_attributes() const
    {
        return base_type::m_num_features;
    }

    /**
     * Returns the total number of labels.
     *  @return int         The total number of labels. This is always 0 for
     *                      the data collection for candidate instances.
     */
    inline int num_labels() const
    {
        return 0;
    }

    /**
     * Returns the total number of features.
     *  @return int         The total number of features.
     */
    int num_features() const
    {
        return base_type::m_num_features;
    }
};




/**
 * A template class for a collection of candidate-classification instances.
 * with quarks assigning label and attribute identifiers.
 *
 *  This class represents a data set for training a candidate classifier. The
 *  class stores instances into a vector. This class implements the necessary
 *  functions num_attributes(), num_features(), and num_labels() for training
 *  algorithms. Do not forget to call set_num_features() to specify the total
 *  number of features.
 *
 *  @param  instance_tmpl           The type of an instance.
 *  @param  attributes_quark_tmpl   The type of an attribute quark.
 *  @param  label_quark_tmpl        The type of a label quark.
 *  @param  features_quark_tmpl     The type of a feature quark.
 */
template <
    class instance_tmpl,
    class attributes_quark_tmpl,
    class labels_quark_tmpl,
    class feature_generator_tmpl
>
class candidate_data_with_quark_base :
    public candidate_data_base<instance_tmpl, feature_generator_tmpl>
{
public:
    /// The type of an instance.
    typedef instance_tmpl instance_type;
    /// The type of a feature vector.
    typedef attributes_quark_tmpl attributes_quark_type;
    /// The type of label quark.
    typedef labels_quark_tmpl labels_quark_type;
    /// The type of the feature-generator class.
    typedef feature_generator_tmpl feature_generator_type;
    /// The base class.
    typedef candidate_data_base<instance_tmpl, feature_generator_tmpl> base_type;

    /// A type providing a container of instances.
    typedef typename base_type::instances_type instances_type;
    /// A type counting the number of pairs in a container.
    typedef typename base_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename base_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename base_type::const_iterator const_iterator;
    /// The type of an attribute.
    typedef typename base_type::attribute_type attribute_type;

    /// The type of a label.
    typedef typename labels_quark_type::value_type label_type;
    /// A set of attributes in the data set.
    attributes_quark_type attributes;
    /// A set of labels in the data set.
    labels_quark_type labels;

public:
    /**
     * Constructs the object.
     */
    candidate_data_with_quark_base()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~candidate_data_with_quark_base()
    {
    }

    /**
     * Returns the total number of attributes.
     *  @return int         The total number of attributes.
     */
    int num_attributes() const
    {
        return attributes.size();
    }

    /**
     * Returns the total number of labels.
     *  @return int         The total number of labels.
     */
    int num_labels() const
    {
        return labels.size();
    }

    /**
     * Returns the total number of features.
     *  @return int         The total number of features.
     */
    int num_features() const
    {
        return attributes.size();
    }
};



/**
 * Data set for classification instances.
 *
 *  This class provides a data set for classification (attribute-label)
 *  instances.
 *
 *  @param  instance_tmpl           The type of an instance.
 *  @param  feature_generator_tmpl  The type of a feature generator.
 */
template <
    class instance_tmpl,
    class feature_generator_tmpl
>
class multi_data_base :
    public candidate_data_base<
        instance_tmpl,
        feature_generator_tmpl
        >
{
public:
    /// The type of an instance.
    typedef instance_tmpl instance_type;
    /// The type of the feature-generator class.
    typedef feature_generator_tmpl feature_generator_type;
 
    /// The base class.
    typedef candidate_data_base<instance_tmpl, feature_generator_tmpl> base_type;
    /// A type providing a container of instances.
    typedef typename base_type::instances_type instances_type;
    /// A type counting the number of pairs in a container.
    typedef typename base_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename base_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename base_type::const_iterator const_iterator;
    /// The type of an attribute.
    typedef typename base_type::attribute_type attribute_type;

    /// The number of features.
    int m_num_labels;

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
     * Returns the total number of attributes.
     *  @return int         The total number of attributes.
     */
    int num_attributes() const
    {
        return base_type::m_num_features;
    }

    /**
     * Returns the total number of labels.
     *  @return int         The total number of labels. This is always 2 for
     *                      the data collection for binary instances.
     */
    int num_labels() const
    {
        return m_num_labels;
    }

    /**
     * Returns the total number of features.
     *  @return int         The total number of features.
     */
    int num_features() const
    {
        return this->feature_generator.num_features();
    }

    void generate_bias_features(const attribute_type& a)
    {
        this->feature_generator.set_num_labels(this->labels.size());
        this->feature_generator.set_num_attributes(this->attributes.size());

        int max = -1;
        for (int l = 0;l < this->num_labels();++l) {
            int fid = this->feature_generator.regist(a, l);
            if (max < fid) {
                max = fid;
            }
        }

        this->set_user_feature_start(max+1);
    }

    /**
     * Finalize the data set.
     */
    void generate_features()
    {
        this->feature_generator.set_num_labels(this->labels.size());
        this->feature_generator.set_num_attributes(this->attributes.size());

        if (this->feature_generator.needs_registration()) {
            iterator iti;
            for (iti = this->begin();iti != this->end();++iti) {
                typename instance_type::iterator it;
                for (it = iti->begin();it != iti->end();++it) {
                    this->feature_generator.regist(it->first, iti->get_label());
                }
            }
        }
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
    class labels_quark_tmpl,
    class feature_generator_tmpl
>
class multi_data_with_quark_base :
    public multi_data_base<instance_tmpl, feature_generator_tmpl>
{
public:
    /// The type of an instance.
    typedef instance_tmpl instance_type;
    /// The type of a feature vector.
    typedef attributes_quark_tmpl attributes_quark_type;
    /// The type of label quark.
    typedef labels_quark_tmpl labels_quark_type;
    /// The type of the feature-generator class.
    typedef feature_generator_tmpl feature_generator_type;
 
    /// The base class.
    typedef multi_data_base<instance_tmpl, feature_generator_tmpl> base_type;
    /// A type providing a container of instances.
    typedef typename base_type::instances_type instances_type;
    /// A type counting the number of pairs in a container.
    typedef typename base_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename base_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename base_type::const_iterator const_iterator;
    /// The type of an attribute.
    typedef typename base_type::attribute_type attribute_type;

    /// The type of a label.
    typedef typename labels_quark_type::value_type label_type;
    /// A set of attributes in the data set.
    attributes_quark_type attributes;
    /// A set of labels in the data set.
    labels_quark_type labels;

public:
    /**
     * Constructs the object.
     */
    multi_data_with_quark_base()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~multi_data_with_quark_base()
    {
    }

    /**
     * Returns the total number of attributes.
     *  @return int         The total number of attributes.
     */
    int num_attributes() const
    {
        return attributes.size();
    }

    /**
     * Returns the total number of labels.
     *  @return int         The total number of labels.
     */
    int num_labels() const
    {
        return labels.size();
    }

    /**
     * Returns the total number of features.
     *  @return int         The total number of features.
     */
    int num_features() const
    {
        return this->feature_generator.num_features();
    }

    void generate_bias_features(const attribute_type& a)
    {
        this->feature_generator.set_num_labels(this->labels.size());
        this->feature_generator.set_num_attributes(this->attributes.size());

        int max = -1;
        for (int l = 0;l < this->num_labels();++l) {
            int fid = this->feature_generator.regist(a, l);
            if (max < fid) {
                max = fid;
            }
        }

        this->set_user_feature_start(max+1);
    }

    /**
     * Finalize the data set.
     */
    void generate_features()
    {
        this->feature_generator.set_num_labels(this->labels.size());
        this->feature_generator.set_num_attributes(this->attributes.size());

        if (this->feature_generator.needs_registration()) {
            iterator iti;
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
