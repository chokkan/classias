#ifndef __CLASSIAS_BASE_H__
#define __CLASSIAS_BASE_H__

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "quark.h"
#include "feature.h"

namespace classias
{

/**
 * Sparse attribute vector.
 *
 *  This class implements a sparse attribute vector as a linear array of
 *  elements, pairs of attribute names and values.
 *
 *  @param  name_base       The type of attribute names.
 *  @param  value_base      The type of attribute values.
 */
template <class name_base, class value_base>
class sparse_attributes_base
{
public:
    /// A type representing an attribute identifier.
    typedef name_base name_type;
    /// A type representing an attribute value.
    typedef value_base value_type;
    /// A type representing an attribute element, a pair of (name, value).
    typedef std::pair<name_type, value_type> attribute_type;
    /// A type providing a container of (name, value) pairs.
    typedef std::vector<attribute_type> container_type;
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
     * Constructs an attribute vector.
     */
    sparse_attributes_base()
    {
    }

    /**
     * Destructs the attribute vector.
     */
    virtual ~sparse_attributes_base()
    {
    }

    /**
     * Erases all the attributes of the vector.
     */
    inline void clear()
    {
        cont.clear();
    }

    /**
     * Tests if the attribute vector is empty.
     *  @retval bool        \c true if the attribute vector is empty,
     *                      \c false otherwise.
     */
    inline bool empty() const
    {
        return cont.empty();
    }

    /**
     * Returns the number of attributes in the vector.
     *  @retval size_type   The current size of the attribute vector.
     */
    inline size_type size() const
    {
        return cont.size();
    }

    /**
     * Returns a random-access iterator to the first attribute.
     *  @retval iterator    A random-access iterator (for read/write)
     *                      addressing the first attribute in the vector or
     *                      to the location succeeding an empty attributes.
     */
    inline iterator begin()
    {
        return cont.begin();
    }

    /**
     * Returns a random-access iterator to the first attribute.
     *  @retval iterator    A random-access iterator (for read-only)
     *                      addressing the first attribute in the vector or
     *                      to the location succeeding an empty attributes. 
     */
    inline const_iterator begin() const
    {
        return cont.begin();
    }

    /**
     * Returns a random-access iterator pointing just beyond the last attribute.
     *  @retval iterator    A random-access iterator (for read/write)
     *                      addressing the end of the attributes.
     */
    inline iterator end()
    {
        return cont.end();
    }

    /**
     * Returns a random-access iterator pointing just beyond the last attribute.
     *  @retval iterator    A random-access iterator (for read-only)
     *                      addressing the end of the attributes.
     */
    inline const_iterator end() const
    {
        return cont.end();
    }

    /**
     * Adds an attribute (name, value) to the end of the vector.
     *  @param  name        The attribute name.
     *  @param  value       The value of the attribute.
     */
    inline void append(const name_type& name, const value_type& value)
    {
        cont.push_back(attribute_type(name, value));
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

    template <class vector_type>
    inline void add(vector_type& v, const double scale) const
    {
        for (const_iterator it = begin();it != end();++it) {
            v[it->first] += scale * (double)it->second;
        }
    }
};

/**
 * A candidate.
 * */
template <
    class attributes_tmpl,
    class label_tmpl
>
class candidate_base : 
    public attributes_tmpl
{
public:
    typedef attributes_tmpl attributes_type;
    typedef label_tmpl label_type;

    bool torf;
    label_type label;

    candidate_base() : torf(false), label(0)
    {
    }

    virtual ~candidate_base()
    {
    }

    inline bool is_true() const
    {
        return torf;
    }
};



/**
 * Candidate class.
 */
template <class candidate_base>
class candidates_base
{
public:
    /// A type representing an attributes.
    typedef candidate_base candidate_type;
    /// A type providing a container of attributes of all candidates.
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

    inline candidate_type& new_element()
    {
        candidates.push_back(candidate_type());
        return candidates.back();
    }
};

/**
 * Group number class.
 */
class group_base
{
protected:
    typedef int group_type;
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

template <class candidate_tmpl>
class ranking_instance_base :
    public candidates_base<candidate_tmpl>,
    public group_base
{
public:
    typedef candidate_tmpl candidate_type;
    typedef candidates_base<candidate_tmpl> candidates_type;

    typedef typename candidate_type::attributes_type attributes_type;
    typedef typename candidate_type::label_type label_type;

    ranking_instance_base()
    {
    }

    virtual ~ranking_instance_base()
    {
    }
};

template <
    class instance_tmpl,
    class attribute_quark_tmpl,
    class label_quark_tmpl
>
class data_base
{
public:
    typedef instance_tmpl instance_type;
    typedef attribute_quark_tmpl attribute_quark_type;
    typedef label_quark_tmpl label_quark_type;

    typedef typename instance_type::label_type label_type;

    typedef std::vector<instance_type> instances_type;
    typedef typename instances_type::size_type size_type;
    typedef typename instances_type::iterator iterator;
    typedef typename instances_type::const_iterator const_iterator;

    typedef std::vector<label_type> positive_labels_type;

    instances_type instances;
    attribute_quark_type attributes;
    label_quark_type labels;
    positive_labels_type positive_labels;

    data_base()
    {
    }

    virtual ~data_base()
    {
    }

    inline void clear()
    {
        instances.clear();
    }

    inline bool empty() const
    {
        return instances.empty();
    }

    inline size_type size() const
    {
        return instances.size();
    }

    inline iterator begin()
    {
        return instances.begin();
    }

    inline const_iterator begin() const
    {
        return instances.begin();
    }

    inline iterator end()
    {
        return instances.end();
    }

    inline const_iterator end() const
    {
        return instances.end();
    }

    inline instance_type& back()
    {
        return instances.back();
    }

    inline instance_type& new_element()
    {
        instances.push_back(instance_type());
        return back();
    }

    inline size_type num_features() const
    {
        return attributes.size();
    }

    inline size_type num_labels() const
    {
        return labels.size();
    }
};

typedef sparse_attributes_base<int, double> sparse_attributes;

typedef candidate_base<sparse_attributes, int> binstance;
typedef data_base<binstance, quark, quark> sbdata;

typedef candidate_base<sparse_attributes, int> rcandidate;
typedef ranking_instance_base<rcandidate> rinstance;
typedef data_base<rinstance, quark, quark> srdata;

};

#endif/*__CLASSIAS_BASE_H__*/
