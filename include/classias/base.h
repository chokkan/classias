#ifndef __CLASSIAS_BASE_H__
#define __CLASSIAS_BASE_H__

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "quark.h"

namespace classias
{



/**
 * Truth class.
 *
 *  This class implements the base interface indicating whether an inherited
 *  object presents a true instance/candidate.
 */
class truth_base
{
protected:
    /// The type representing the truth.
    typedef bool truth_type;
    /// The truth value.
    truth_type m_truth;

public:
    /**
     * Constructs the object.
     */
    truth_base() : m_truth(false)
    {
    }

    /**
     * Constructs the object.
     *  @param  truth           The truth.
     */
    truth_base(const truth_type& truth) : m_truth(truth)
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~truth_base()
    {
    }

    /**
     * Set the truth.
     *  @param  truth       The truth.
     */
    inline void set_truth(truth_type truth)
    {
        m_truth = truth;
    }

    /**
     * Get the truth.
     *  @retval truth_type  The truth.
     */
    inline int get_truth() const
    {
        return m_truth;
    }
};





/**
 * Group number class.
 *
 * The class implements the base interface for instances with group numbers.
 */
class group_base
{
protected:
    /// A type representing a group number.
    typedef int group_type;
    /// The group number.
    group_type m_group;

public:
    /**
     * Constructs the object.
     */
    group_base() : m_group(0)
    {
    }

    /**
     * Constructs the object.
     *  @param  group           The group number.
     */
    group_base(const group_type& group) : m_group(group)
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~group_base()
    {
    }

    /**
     * Set the group number.
     *  @param  group       The group number.
     */
    inline void set_group(int group)
    {
        m_group = group;
    }

    /**
     * Get the group number.
     *  @retval int         The group number.
     */
    inline int get_group() const
    {
        return m_group;
    }
};





/**
 * Sparse vector.
 *
 *  This class implements a sparse vector as a linear array of elements, pairs
 *  of names and values.
 *
 *  @param  name_base       The type of element names.
 *  @param  value_base      The type of element values.
 */
template <class name_base, class value_base>
class sparse_vector_base
{
public:
    /// A type representing an element name.
    typedef name_base name_type;
    /// A type representing an element value.
    typedef value_base value_type;
    /// A type representing an element, a pair of (name, value).
    typedef std::pair<name_type, value_type> element_type;
    /// A type providing a container of (name, value) pairs.
    typedef std::vector<element_type> container_type;
    /// A type counting the number of pairs in a container.
    typedef typename container_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename container_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename container_type::const_iterator const_iterator;

protected:
    /// A container of (name, value) pairs.
    container_type cont;

public:
    /**
     * Constructs a sparse vector.
     */
    sparse_vector_base()
    {
    }

    /**
     * Destructs the sparse vector.
     */
    virtual ~sparse_vector_base()
    {
    }

    /**
     * Erases all the elements of the vector.
     */
    inline void clear()
    {
        cont.clear();
    }

    /**
     * Tests if the sparse vector is empty.
     *  @retval bool        \c true if the sparse vector is empty,
     *                      \c false otherwise.
     */
    inline bool empty() const
    {
        return cont.empty();
    }

    /**
     * Returns the number of elements in the vector.
     *  @retval size_type   The current size of the sparse vector.
     */
    inline size_type size() const
    {
        return cont.size();
    }

    /**
     * Returns a random-access iterator to the first element.
     *  @retval iterator    A random-access iterator (for read/write)
     *                      addressing the first element in the vector or
     *                      to the location succeeding an empty element.
     */
    inline iterator begin()
    {
        return cont.begin();
    }

    /**
     * Returns a random-access iterator to the first element.
     *  @retval iterator    A random-access iterator (for read-only)
     *                      addressing the first element in the vector or
     *                      to the location succeeding an empty element. 
     */
    inline const_iterator begin() const
    {
        return cont.begin();
    }

    /**
     * Returns a random-access iterator pointing just beyond the last element.
     *  @retval iterator    A random-access iterator (for read/write)
     *                      addressing the end of the element.
     */
    inline iterator end()
    {
        return cont.end();
    }

    /**
     * Returns a random-access iterator pointing just beyond the last element.
     *  @retval iterator    A random-access iterator (for read-only)
     *                      addressing the end of the element.
     */
    inline const_iterator end() const
    {
        return cont.end();
    }

    /**
     * Adds an element (name, value) to the end of the vector.
     *  @param  name        The element name.
     *  @param  value       The element value.
     */
    inline void append(const name_type& name, const value_type& value)
    {
        cont.push_back(element_type(name, value));
    }

    /**
     * Compute the inner product with another vector.
     *  @param  v           The vector.
     *  @retval double      The inner product.
     */
    template <class vector_type>
    inline double inner_product(const vector_type& v) const
    {
        double s = 0.;
        for (const_iterator it = begin();it != end();++it) {
            s += (double)v[it->first] * (double)it->second;
        }
        return s;
    }

    /**
     * Add the scaled value to another vector.
     *  @param  v           The vector to which this function adds the value.
     *  @param  scale       The scale factor.
     */
    template <class vector_type>
    inline void add_to(vector_type& v, const double scale) const
    {
        for (const_iterator it = begin();it != end();++it) {
            v[it->first] += scale * (double)it->second;
        }
    }
};





/**
 * Ranking candidate.
 *
 *  This class represents a ranking candidate that consists of a feature
 *  vector, truth, and label.
 *
 *  @param  features_tmpl   The type of feature vector.
 *  @param  label_tmpl      The type of candidate label.
 */
template <
    class features_tmpl,
    class label_tmpl
>
class multi_candidate_base : 
    public features_tmpl,
    public truth_base
{
public:
    /// The type of a feature vector.
    typedef features_tmpl features_type;
    /// The type of a candidate label.
    typedef label_tmpl label_type;

    /// The candidate label.
    label_type m_label;

    /**
     * Constructs a candidate.
     */
    multi_candidate_base() : m_label(0)
    {
    }

    /**
     * Destructs a candidate.
     */
    virtual ~multi_candidate_base()
    {
    }

    /**
     * Set the candidate label.
     *  @param  label       The candidate label.
     */
    inline void set_label(label_type label)
    {
        m_label = label;
    }

    /**
     * Get the candidate label.
     *  @retval  label_type The candidate label.
     */
    inline label_type get_label() const
    {
        return m_label;
    }
};





/**
 * Ranking candidates.
 *
 *  This class implements a linear array of ranking candidates.
 */
template <class multi_candidate_base>
class candidates_base
{
public:
    /// A type representing an features.
    typedef multi_candidate_base candidate_type;
    /// A type providing a container of features of all candidates.
    typedef std::vector<candidate_type> candidates_type;
    /// A type counting the number of candidates in the instance.
    typedef typename candidates_type::size_type size_type;
    /// A type providing a random-access iterator for candidates.
    typedef typename candidates_type::iterator iterator;
    /// A type providing a read-only random-access iterator for candidates.
    typedef typename candidates_type::const_iterator const_iterator;

protected:
    /// A container of all candidates associated with the instance.
    candidates_type candidates;

public:
    /**
     * Constructs an object.
     */
    candidates_base()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~candidates_base()
    {
    }

    /**
     * Erases all the candidates in the object.
     */
    inline void clear()
    {
        candidates.clear();
    }

    /**
     * Tests if the object has no candidate.
     *  @retval bool        \c true if the object has no candidate,
     *                      \c false otherwise.
     */
    inline bool empty() const
    {
        return candidates.empty();
    }

    /**
     * Returns the number of candidates.
     *  @retval int     The number of candidates associated with the object.
     */
    inline size_t size() const
    {
        return candidates.size();
    }

    /**
     * Returns a random-access iterator to the first candidate.
     *  @retval iterator    A random-access iterator (for read/write)
     *                      addressing the first candidate or to the location
     *                      succeeding an empty candidate.
     */
    inline iterator begin()
    {
        return candidates.begin();
    }

    /**
     * Returns a random-access iterator to the first candidate.
     *  @retval iterator    A random-access iterator (for read-only)
     *                      addressing the first candidate or to the location
     *                      succeeding an empty candidate.
     */
    inline const_iterator begin() const
    {
        return candidates.begin();
    }

    /**
     * Returns a random-access iterator pointing just beyond the last candidate.
     *  @retval iterator    A random-access iterator (for read/write)
     *                      addressing the end of the candidates.
     */
    inline iterator end()
    {
        return candidates.end();
    }

    /**
     * Returns a random-access iterator pointing just beyond the last candidate.
     *  @retval iterator    A random-access iterator (for read-only)
     *                      addressing the end of the candidates.
     */
    inline const_iterator end() const
    {
        return candidates.end();
    }

    /**
     * Adds an candidate to the object.
     *  @param  candidate   The candidate to be inserted to this object.
     */
    inline void append(const candidate_type& candidate)
    {
        candidates.push_back(candidate);
    }

    /**
     * Create a new candidate.
     *  @retval candidate_type& The reference to the new candidate.
     */
    inline candidate_type& new_element()
    {
        candidates.push_back(candidate_type());
        return candidates.back();
    }
};



/**
 * Binary instance.
 *
 *  This class implements a binary-classification instance that consists of
 *  a feature vector, truth, and group number.
 *
 *  @param  features_tmpl   The type of feature vector.
 */
template <class features_tmpl>
class binary_instance_base :
    public features_tmpl,
    public truth_base,
    public group_base
{
public:
    /// The type of a feature vector.
    typedef features_tmpl features_type;

    /**
     * Constructs an object.
     */
    binary_instance_base()
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~binary_instance_base()
    {
    }
};





/**
 * Multi-candidate instance.
 *  
 *  This class implements a multi-candidate instance that consists of
 *  an array of candidates and group number.

 */
template <class candidate_tmpl>
class multi_instance_base :
    public candidates_base<candidate_tmpl>,
    public group_base
{
public:
    /// The type of a candidate.
    typedef candidate_tmpl candidate_type;
    /// The type of multiple candidates.
    typedef candidates_base<candidate_tmpl> candidates_type;
    /// The type of a feature vector.
    typedef typename candidate_type::features_type features_type;
    /// The type of a candidate label.
    typedef typename candidate_type::label_type label_type;

    /**
     * Constructs an object.
     */
    multi_instance_base()
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~multi_instance_base()
    {
    }
};



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

    /// A container of instances.
    instances_type instances;
    /// A feature quark.
    features_quark_type features;
    /// The start index of features.
    feature_type feature_end_index;

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
        instances.push_back(instance_type());
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

    /**
     * Returns the number of features in the data.
     *  @retval size_type   The number of features.
     */
    inline size_type num_features() const
    {
        return features.size();
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
    typedef std::vector<label_type> positive_labels_type;

    label_quark_type labels;
    positive_labels_type positive_labels;

    multi_data_base()
    {
    }

    virtual ~multi_data_base()
    {
    }
};

typedef sparse_vector_base<int, double> sparse_attributes;

typedef binary_instance_base<sparse_attributes> binstance;
typedef binary_data_base<binstance, quark> bdata;

typedef multi_candidate_base<sparse_attributes, int> mcandidate;
typedef multi_instance_base<mcandidate> minstance;
typedef multi_data_base<minstance, quark, quark> mdata;

};

#endif/*__CLASSIAS_BASE_H__*/
