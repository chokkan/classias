#ifndef __CLASSIAS_BASE_H__
#define __CLASSIAS_BASE_H__

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace classias
{

class unknown_association : public std::out_of_range
{
public:
    unknown_association(const std::string& message) : std::out_of_range(message)
    {
    }
};

template <class item_base>
class basic_quark {
public:
    typedef item_base item_type;

    typedef std::vector<item_type> inverse_map_type;
    typedef typename inverse_map_type::size_type value_type;
    typedef std::map<item_type, value_type> forward_map_type;

protected:
    /// Forward mapping: (item0, item1) -> value.
    forward_map_type m_fwd;
    /// Inverse mapping: value -> (item0, item1).
    inverse_map_type m_inv;

public:
    basic_quark()
    {
    }

    virtual ~basic_quark()
    {
    }

    inline value_type size() const
    {
        return m_fwd.size();
    }

    inline bool exists(const item_type& x)
    {
        return m_fwd.find(x) != m_fwd.end();
    }

    inline value_type operator() (const item_type& x)
    {
        return associate(x);
    }

    inline value_type associate(const item_type& x)
    {
        typename forward_map_type::const_iterator it = m_fwd.find(x);
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            value_type v = m_inv.size();
            m_fwd.insert(typename forward_map_type::value_type(x, v));
            m_inv.push_back(x);
            return v;
        }
    }

    inline value_type to_value(const item_type& x) const
    {
        typename forward_map_type::const_iterator it = m_fwd.find(x);
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            throw unknown_association("Unknown forward mapping");
        }           
    }

    inline const item_type& to_item(const value_type& v) const
    {
        if (v < m_inv.size()) {
            return m_inv[v];
        } else {
            throw unknown_association("Unknown inverse mapping");
        }
    }
};



template <class item0_base, class item1_base>
class basic_quark2 {
public:
    typedef item0_base item0_type;
    typedef item1_base item1_type;

    typedef std::pair<item0_type, item1_type> elem_type;
    typedef std::vector<elem_type> inverse_map_type;
    typedef typename inverse_map_type::size_type value_type;
    typedef std::map<elem_type, value_type> forward_map_type;

protected:
    /// Forward mapping: (item0, item1) -> value.
    forward_map_type m_fwd;
    /// Inverse mapping: value -> (item0, item1).
    inverse_map_type m_inv;

public:
    basic_quark2()
    {
    }

    virtual ~basic_quark2()
    {
    }

    inline value_type size() const
    {
        return m_fwd.size();
    }

    inline value_type operator() (const item0_type& x, const item1_base& y)
    {
        return associate(x, y);
    }

    inline bool exists(const item0_type& x, const item1_base& y)
    {
        return m_fwd.find(elem_type(x, y)) != m_fwd.end();
    }

    inline value_type associate(const item0_type& x, const item1_base& y)
    {
        typename forward_map_type::const_iterator it = m_fwd.find(elem_type(x, y));
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            value_type v = m_inv.size();
            elem_type il(x, y);
            m_fwd.insert(typename forward_map_type::value_type(il, v));
            m_inv.push_back(il);
            return v;
        }
    }

    inline value_type to_value(const item0_type& x, const item1_type& y) const
    {
        typename forward_map_type::const_iterator it = m_fwd.find(elem_type(x, y));
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            throw unknown_association("Unknown forward mapping");
        }           
    }

    inline void to_item(const value_type& v, item0_type& x, item1_base& y) const
    {
        if (v < m_inv.size()) {
            x = m_inv[v].first;
            y = m_inv[v].second;
        } else {
            throw unknown_association("Unknown inverse mapping");
        }
    }
};

template <class label_temp>
class label_base
{
protected:
    enum {
        flag_true = 0x10000000,
        flag_positive = 0x20000000,
    };

public:
    typedef label_temp label_type;
    label_type label;

    label_base()
    {
    }

    label_base(const label_type& l) : label(l)
    {
    }

    virtual ~label_base()
    {
    }

    inline void set_label(const label_type& l)
    {
        label = l;
    }

    inline const label_type& get_label() const
    {
        return label;
    }

    inline void set_label(bool is_true, bool is_positive)
    {
        label = 0;
        if (is_true) label += flag_true;
        if (is_positive) label += flag_positive;
    }

    inline bool is_true() const
    {
        return (label & flag_true);
    }

    inline bool is_positive() const
    {
        return (label & flag_positive);
    }
};


/**
 * Sparse attribute vector.
 *
 *  This class implements a sparse attribute vector as an linear array of
 *  elements (pairs of attribute names and values).
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
    inline void add(vector_type& v, double scale) const
    {
        for (const_iterator it = begin();it != end();++it) {
            v[it->first] += scale * (double)it->second;
        }
    }
};

template <class attributes_tmpl, class label_tmpl>
class labeled_attributes_base : 
    public attributes_tmpl,
    public label_base<label_tmpl>
{
public:
    typedef attributes_tmpl attributes_type;
    typedef label_tmpl label_type;

    labeled_attributes_base()
    {
    }

    labeled_attributes_base(const label_type& l) :
        label_base<label_tmpl>(l)
    {
    }

    virtual ~labeled_attributes_base()
    {
    }
};

/**
 * Instance class.
 */
template <class candidate_base>
class candidates_base
{
public:
    /// A type representing a attributes.
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
     * Constructs an instance.
     */
    candidates_base()
    {
    }

    /**
     * Destructs an instance.
     */
    virtual ~candidates_base()
    {
    }

    /**
     * Erases all the candidates in the instance.
     */
    inline void clear()
    {
        candidates.clear();
    }

    /**
     * Tests if the instance is empty.
     *  @retval bool        \c true if the instance is empty,
     *                      \c false otherwise.
     */
    inline bool empty() const
    {
        return candidates.empty();
    }

    /**
     * Returns the number of candidates for the instance.
     *  @retval int     The number of candidates associated with the instance.
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
     * Adds an attribute (id, value) to the end of the vector.
     *  @param  id          An identifier.
     *  @param  value       A value.
     */
    inline void append(const candidate_type& candidate)
    {
        candidates.push_back(candidate);
    }

    inline candidate_type& new_element()
    {
        candidates.resize(candidates.size()+1);
        return candidates.back();
    }
};

class group_base
{
public:
    typedef int group_type;
    group_type group;

    group_base() : group(0)
    {
    }

    group_base(const group_type& _group) : group(_group)
    {
    }

    virtual ~group_base()
    {
    }

    inline void set_group(int g)
    {
        group = g;
    }

    inline int get_group() const
    {
        return group;
    }
};

template <class instance_tmpl>
class classification_instance_base :
    public instance_tmpl,
    public group_base
{
public:
    classification_instance_base()
    {
    }

    virtual ~classification_instance_base()
    {
    }
};

template <class instance_tmpl>
class selection_instance_base :
    public classification_instance_base<instance_tmpl>
{
public:
    typedef typename instance_tmpl::label_type label_type;
    typedef candidates_base<label_base<label_type> > labels_type;
    labels_type labels;

    selection_instance_base()
    {
    }

    virtual ~selection_instance_base()
    {
    }
};

template <class instance_tmpl>
class ranking_instance_base :
    public candidates_base<instance_tmpl>,
    public group_base
{
public:
    ranking_instance_base()
    {
    }

    virtual ~ranking_instance_base()
    {
    }
};




typedef basic_quark<int> quark;
typedef basic_quark2<int, int> quark2;
typedef sparse_attributes_base<int, double> sparse_attributes;
typedef labeled_attributes_base<sparse_attributes, int> instance;
typedef classification_instance_base<instance> cinstance;
typedef selection_instance_base<instance> sinstance;
typedef ranking_instance_base<instance> rinstance;

typedef basic_quark<std::string> string_quark;
typedef basic_quark2<std::string, std::string> string_quark2;


};

#endif/*__CLASSIAS_BASE_H__*/
