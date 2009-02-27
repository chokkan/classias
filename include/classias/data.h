#ifndef __CLASSIAS_DATA_H__
#define __CLASSIAS_DATA_H__

namespace classias
{

/**
 * Data set for binary-classification instances.
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
    typedef typename instance_type::traits_type data_traits_type;

    /// A container of instances.
    instances_type instances;
    /// A feature quark.
    features_quark_type features;
    /// The start index of features.
    feature_type feature_end_index;
    ///
    data_traits_type traits;

    /**
     * Constructs the object.
     */
    binary_data_base() : feature_end_index(0)
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
     * Create a new instance.
     *  @retval instance_type&  The reference to the new instance.
     */
    inline instance_type& new_element()
    {
        instances.push_back(instance_type(&traits));
        return this->back();
    }

    inline void set_user_feature_end(feature_type index)
    {
        feature_end_index = index;
    }

    inline feature_type get_user_feature_end() const
    {
        return feature_end_index;
    }

    void finalize()
    {
        traits.set_num_labels(2);
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
class multi_data_base : public binary_data_base<instance_tmpl, features_quark_tmpl>
{
public:
    typedef label_quark_tmpl label_quark_type;
    typedef typename label_quark_type::value_type label_type;
    typedef typename binary_data_base<instance_tmpl, features_quark_tmpl>::size_type size_type;
    typedef std::vector<label_type> positive_labels_type;

    label_quark_type labels;
    positive_labels_type positive_labels;

    multi_data_base()
    {
    }

    virtual ~multi_data_base()
    {
    }

    void append_positive_label(label_type l)
    {
        positive_labels.push_back(l);
    }

    void finalize()
    {
        this->traits.set_num_labels(this->labels.size());
        this->traits.set_num_attributes(this->features.size());
    }
};


template <
    class instance_tmpl,
    class attributes_quark_tmpl,
    class label_quark_tmpl
>
class attribute_data_base :
    public multi_data_base<instance_tmpl, attributes_quark_tmpl, label_quark_tmpl>
{
public:
    typedef multi_data_base<instance_tmpl, attributes_quark_tmpl, label_quark_tmpl> base_type; 
    typedef typename base_type::instance_type instance_type;

public:
    attribute_data_base()
    {
    }

    virtual ~attribute_data_base()
    {
    }

    void finalize()
    {
        this->traits.set_num_labels(this->labels.size());
        this->traits.set_num_attributes(this->features.size());

        if (this->traits.needs_generate()) {
            typename base_type::iterator it;
            for (it = this->begin();it != this->end();++it) {
                it->generate();
            }
        }
    }
};

};

#endif/*__CLASSIAS_DATA_H__*/
