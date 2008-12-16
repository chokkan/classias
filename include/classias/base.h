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
public:
    /// The type representing the truth.
    typedef bool truth_type;

protected:
    /// The truth value.
    truth_type m_truth;

public:
    /**
     * Constructs an object.
     */
    truth_base() : m_truth(false)
    {
    }

    /**
     * Constructs an object initialized by the specified value.
     *  @param  truth       The truth value used to initialize the object.
     */
    truth_base(const truth_type& truth) : m_truth(truth)
    {
    }

    /**
     * Constructs an object that is a copy of some other object.
     *  @param  rho         The original object used to initialize the object.
     */
    truth_base(const truth_base& rho) : m_truth(rho.m_truth)
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~truth_base()
    {
    }

    /**
     * Assigns a new truth value to the object.
     *  @param  rho         The source object.
     *  @retval truth_base& The reference to this object.
     */
    truth_base& operator=(const truth_base& rho)
    {
        m_truth = rho.m_truth;
        return *this;
    }

    /**
     * Tests the equality of two truth objects.
     *  @param  x           A truth object.
     *  @param  y           Another truth object.
     *  @retval bool        \c true if the values of two objects are identical,
     *                      \c false otherwise.
     */
    inline friend bool operator==(
        const truth_base& x,
        const truth_base& y
        )
    {
        return (x.m_truth == y.m_truth);
    }

    /**
     * Tests the inequality of two truth objects.
     *  @param  x           A truth object.
     *  @param  y           Another truth object.
     *  @retval bool        \c true if the values of two objects are not
     *                      identical, \c false otherwise.
     */
    inline friend bool operator!=(
        const truth_base& x,
        const truth_base& y
        )
    {
        return (x.m_truth != y.m_truth);
    }

    /**
     * Assigns a new truth value to the object.
     *  @param  truth       The truth value.
     */
    inline void set_truth(truth_type truth)
    {
        m_truth = truth;
    }

    /**
     * Obtains the current truth value.
     *  @retval truth_type  The current value.
     */
    inline truth_type get_truth() const
    {
        return m_truth;
    }
};



/**
 * Label class.
 *
 *  This class implements the base interface to represent a label of an
 *  instance/candidate.
 */
template <class label_tmpl>
class label_base
{
public:
    /// The type representing the label.
    typedef label_tmpl label_type;
protected:
    /// The label value.
    label_type m_label;

public:
    /**
     * Constructs the object.
     */
    label_base()
    {
    }

    /**
     * Constructs an object initialized by the specified value.
     *  @param  label       The label used to initialize the object.
     */
    label_base(const label_type& label) : m_label(label)
    {
    }

    /**
     * Constructs an object that is a copy of some other object.
     *  @param  rho         The original object used to initialize the object.
     */
    label_base(const label_base& rho) : m_label(rho.m_label)
    {
    }

    /**
     * Assigns a new label to the object.
     *  @param  rho         The source object.
     *  @retval label_base& The reference to this object.
     */
    label_base& operator=(const label_base& rho)
    {
        m_label = rho.m_label;
        return *this;
    }

    /**
     * Destructs the object.
     */
    virtual ~label_base()
    {
    }

    /**
     * Tests the equality of two labelsl.
     *  @param  x           A label.
     *  @param  y           Another label.
     *  @retval bool        \c true if the two labels are identical, \c false
     *                      otherwise.
     */
    inline friend bool operator==(
        const label_base& x,
        const label_base& y
        )
    {
        return (x.m_label == y.m_label);
    }

    /**
     * Tests the inequality of two labels.
     *  @param  x           A label.
     *  @param  y           Another label.
     *  @retval bool        \c true if the two labels are not identical,
     *                      \c false otherwise.
     */
    inline friend bool operator!=(
        const label_base& x,
        const label_base& y
        )
    {
        return (x.m_label != y.m_label);
    }

    /**
     * Assigns a new label to the object.
     *  @param  label       The label.
     */
    inline void set_label(const label_type& label)
    {
        m_label = label;
    }

    /**
     * Obtains the current label.
     *  @retval label_type  The current label.
     */
    inline const label_type& get_label() const
    {
        return m_label;
    }
};



class weight_base
{
public:
    /// The type representing the weight.
    typedef double weight_type;

protected:
    /// The truth value.
    weight_type m_weight;

public:
    /**
     * Constructs an object.
     */
    weight_base() : m_weight(1.)
    {
    }

    /**
     * Constructs an object initialized by the specified value.
     *  @param  truth       The truth value used to initialize the object.
     */
    weight_base(const weight_type& weight) : m_weight(weight)
    {
    }

    /**
     * Constructs an object that is a copy of some other object.
     *  @param  rho         The original object used to initialize the object.
     */
    weight_base(const weight_base& rho) : m_weight(rho.m_weight)
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~weight_base()
    {
    }

    /**
     * Assigns a new truth value to the object.
     *  @param  rho         The source object.
     *  @retval truth_base& The reference to this object.
     */
    weight_base& operator=(const weight_base& rho)
    {
        m_weight = rho.m_weight;
        return *this;
    }

    /**
     * Tests the equality of two truth objects.
     *  @param  x           A truth object.
     *  @param  y           Another truth object.
     *  @retval bool        \c true if the values of two objects are identical,
     *                      \c false otherwise.
     */
    inline friend bool operator==(
        const weight_base& x,
        const weight_base& y
        )
    {
        return (x.m_weight == y.m_weight);
    }

    /**
     * Tests the inequality of two truth objects.
     *  @param  x           A truth object.
     *  @param  y           Another truth object.
     *  @retval bool        \c true if the values of two objects are not
     *                      identical, \c false otherwise.
     */
    inline friend bool operator!=(
        const weight_base& x,
        const weight_base& y
        )
    {
        return (x.m_weight != y.m_weight);
    }

    /**
     * Assigns a new truth value to the object.
     *  @param  truth       The truth value.
     */
    inline void set_weight(weight_type weight)
    {
        m_weight = weight;
    }

    /**
     * Obtains the current truth value.
     *  @retval truth_type  The current value.
     */
    inline weight_type get_weight() const
    {
        return m_weight;
    }
};



/**
 * Group number class.
 *
 * The class implements the base interface for instances with group numbers.
 */
class group_base
{
public:
    /// A type representing a group number.
    typedef int group_type;

protected:
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
     * Constructs an object initialized by the specified number.
     *  @param  group       The group number used to initialize the object.
     */
    group_base(const group_type& group) : m_group(group)
    {
    }

    /**
     * Constructs an object that is a copy of some other object.
     *  @param  rho         The original object used to initialize the object.
     */
    group_base(const group_base& rho) : m_group(rho.m_group)
    {
    }

    /**
     * Assigns a new group number to the object.
     *  @param  rho         The source object.
     *  @retval group_base& The reference to this object.
     */
    group_base& operator=(const group_base& rho)
    {
        m_group = rho.m_group;
        return *this;
    }

    /**
     * Destructs the object.
     */
    virtual ~group_base()
    {
    }

    /**
     * Tests the equality of two group numbers.
     *  @param  x           A group object.
     *  @param  y           Another group object.
     *  @retval bool        \c true if the two group numbers are identical,
     *                      \c false otherwise.
     */
    inline friend bool operator==(
        const group_base& x,
        const group_base& y
        )
    {
        return (x.m_group == y.m_group);
    }

    /**
     * Tests the inequality of two groups.
     *  @param  x           A group object.
     *  @param  y           Another group object.
     *  @retval bool        \c true if the two group numbers are not identical,
     *                      \c false otherwise.
     */
    inline friend bool operator!=(
        const group_base& x,
        const group_base& y
        )
    {
        return (x.m_group != y.m_group);
    }

    /**
     * Assigns a new group number to the object.
     *  @param  group       The group number.
     */
    inline void set_group(group_type group)
    {
        m_group = group;
    }

    /**
     * Obtains the current group number.
     *  @retval group_type  The current group number.
     */
    inline group_type get_group() const
    {
        return m_group;
    }
};



class comment_base
{
public:
    /// A type representing a comment.
    typedef std::string comment_type;

protected:
    /// The comment.
    comment_type m_comment;

public:
    /**
     * Constructs the object.
     */
    comment_base()
    {
    }

    /**
     * Constructs an object initialized by the specified comment.
     *  @param  comment     The comment used to initialize the object.
     */
    comment_base(const comment_type& comment) : m_comment(comment)
    {
    }

    /**
     * Constructs an object that is a copy of some other object.
     *  @param  rho         The original object used to initialize the object.
     */
    comment_base(const comment_base& rho) : m_comment(rho.m_comment)
    {
    }

    /**
     * Assigns a new comment to the object.
     *  @param  rho             The source object.
     *  @retval comment_base&   The reference to this object.
     */
    comment_base& operator=(const comment_base& rho)
    {
        m_comment = rho.m_comment;
        return *this;
    }

    /**
     * Destructs the object.
     */
    virtual ~comment_base()
    {
    }

    /**
     * Tests the equality of two comments.
     *  @param  x           A comment object.
     *  @param  y           Another comment object.
     *  @retval bool        \c true if the two comments are identical,
     *                      \c false otherwise.
     */
    inline friend bool operator==(
        const comment_base& x,
        const comment_base& y
        )
    {
        return (x.m_comment == y.m_comment);
    }

    /**
     * Tests the inequality of two comment.
     *  @param  x           A comment object.
     *  @param  y           Another comment object.
     *  @retval bool        \c true if the two comments are not identical,
     *                      \c false otherwise.
     */
    inline friend bool operator!=(
        const comment_base& x,
        const comment_base& y
        )
    {
        return (x.m_comment != y.m_comment);
    }

    /**
     * Assigns a new comment to the object.
     *  @param  comment     The comment.
     */
    inline void set_comment(const comment_type& comment)
    {
        m_comment = comment;
    }

    /**
     * Obtains the current comment.
     *  @retval group_type  The current comment.
     */
    inline const comment_type& get_comment() const
    {
        return m_comment;
    }
};



    
/**
 * Sparse vector.
 *
 *  This class implements a sparse vector as a linear array of elements, pairs
 *  of identifiers and values.
 *
 *  @param  identifier_base The type of element identifier.
 *  @param  value_base      The type of element values.
 */
template <class identifier_base, class value_base>
class sparse_vector_base
{
public:
    /// A type representing an element identifier.
    typedef identifier_base identifier_type;
    /// A type representing an element value.
    typedef value_base value_type;
    /// A type representing an element, a pair of (identifier, value).
    typedef std::pair<identifier_type, value_type> element_type;
    /// A type providing a container of (identifier, value) pairs.
    typedef std::vector<element_type> container_type;
    /// A type counting the number of pairs in a container.
    typedef typename container_type::size_type size_type;
    /// A type providing a random-access iterator.
    typedef typename container_type::iterator iterator;
    /// A type providing a read-only random-access iterator.
    typedef typename container_type::const_iterator const_iterator;

protected:
    /// A container of (identifier, value) pairs.
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
     *  @param  id          The element identifier.
     *  @param  value       The element value.
     */
    inline void append(const identifier_type& id, const value_type& value)
    {
        cont.push_back(element_type(id, value));
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
 * Labaled candidate class.
 */
template <class instance_tmpl, class label_tmpl>
class labeled_candidate_base
{
public:
    typedef instance_tmpl instance_type;
    typedef label_tmpl label_type;
    typedef typename instance_type::attributes_type attributes_type;
    typedef typename attributes_type::identifier_type attribute_identifier_type;

    const instance_type* m_instance;
    attribute_identifier_type m_offset;
    label_type m_label;

    labeled_candidate_base()
        : m_instance(NULL), m_offset(0), m_label(0)
    {
    }

    labeled_candidate_base(const instance_type* inst)
        : m_instance(inst), m_offset(0), m_label(0)
    {
    }

    labeled_candidate_base(
        const instance_type* inst,
        const label_type& label
        )
        : m_instance(inst), m_offset(0), m_label(label)
    {
        set_label(label);
    }

    labeled_candidate_base(
        const labeled_candidate_base& rho
        )
    {
        operator=(rho);
    }

    inline labeled_candidate_base& operator=(
        const labeled_candidate_base& rho
        )
    {
        m_instance = rho.m_instance;
        m_label = rho.m_label;
        return *this;
    }

    inline friend bool operator==(
        const labeled_candidate_base& x,
        const labeled_candidate_base& y
        )
    {
        return (x.m_instance == y.m_instance && x.m_label == y.m_label);
    }

    inline friend bool operator!=(
        const labeled_candidate_base& x,
        const labeled_candidate_base& y
        )
    {
        return !operator==(x, y);
    }

    /**
     * Set the label.
     *  @param  label       The label.
     */
    inline void set_label(const label_type& label)
    {
        m_offset = m_instance->m_traits->num_attributes() * label;
        m_label = label;
    }

    inline const label_type& get_label() const
    {
        return m_label;
    }

    /**
     * Get the truth.
     *  @retval truth_type  The truth.
     */
    inline bool get_truth() const
    {
        if (m_instance != NULL) {
            return (get_label() == m_instance->get_label());
        } else {
            return false;
        }
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
        typedef typename attributes_type::const_iterator const_iterator;
        const attributes_type& attributes = m_instance->attributes;
        for (const_iterator it = attributes.begin();it != attributes.end();++it) {
            s += (double)v[m_offset + it->first] * (double)it->second;
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
        typedef typename attributes_type::const_iterator const_iterator;
        const attributes_type& attributes = m_instance->attributes;
        for (const_iterator it = attributes.begin();it != attributes.end();++it) {
            v[m_offset + it->first] += scale * (double)it->second;
        }
    }
};







class data_traits
{
public:
    enum {
        DT_NONE,
        DT_BINARY,
        DT_MULTI,
        DT_ATTRIBUTE,
    };

protected:
    int m_data_type;
    int m_num_labels;
    int m_num_attributes;
    int m_num_features;

public:
    data_traits() :
        m_data_type(DT_NONE),
        m_num_labels(0), m_num_attributes(0),
        m_num_features(0)
    {
    }

    virtual ~data_traits()
    {
    }

    int data_type() const
    {
        return m_data_type;
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
        return m_num_features;
    }
};


template <
    class attributes_tmpl,
    class label_tmpl,
    class traits_tmpl
>
class attribute_instance_base :
    public group_base,
    public label_base<label_tmpl>
{
public:
    typedef attributes_tmpl attributes_type;
    typedef typename label_base<label_tmpl>::label_type label_type;
    typedef typename attributes_type::identifier_type attribute_name_type;
    typedef traits_tmpl traits_type;
    typedef attribute_instance_base<attributes_type, label_type, traits_type> instance_type;

    typedef labeled_candidate_base<instance_type, label_type> candidate_type;

    attributes_type attributes;

public:
    const traits_type* m_traits;

    class candidate_iterator
    {
    public:
        candidate_type candidate;

        candidate_iterator()
        {
        }

        candidate_iterator(
            const instance_type* instance,
            label_type l)
            : candidate(instance, l)
        {
        }

        inline candidate_iterator& operator=(const candidate_iterator& rho)
        {
            candidate = rho.candidate;
            return *this;
        }

        inline candidate_type& operator*() const
        {
            return candidate;
        }

        inline const candidate_type* operator->() const
        {
            return &candidate;
        }

        inline candidate_iterator& operator++()
        {
            candidate.set_label(candidate.get_label()+1);
            return *this;
        }

        inline candidate_iterator& operator--()
        {
            candidate.set_label(candidate.get_label()-1);
            return *this;
        }

        inline bool operator==(const candidate_iterator& x)
        {
            return (candidate == x.candidate);
        }

        inline bool operator!=(const candidate_iterator& x)
        {
            return !operator==(x);
        }
    };

    typedef candidate_iterator const_iterator;

public:
    attribute_instance_base() : m_traits(NULL)
    {
    }

    attribute_instance_base(const traits_type* traits)
        : m_traits(traits)
    {
    }

    virtual ~attribute_instance_base()
    {
    }

    inline const_iterator begin() const
    {
        return candidate_iterator(this, 0);
    }

    inline const_iterator end() const
    {
        return candidate_iterator(this, m_traits->num_labels());
    }

    inline label_type size() const
    {
        return m_traits->num_labels();
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
    public truth_base,
    public label_base<label_tmpl>
{
public:
    /// The type of a feature vector.
    typedef features_tmpl features_type;

    /**
     * Constructs a candidate.
     */
    multi_candidate_base()
    {
    }

    /**
     * Destructs a candidate.
     */
    virtual ~multi_candidate_base()
    {
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
    public weight_base,
    public group_base,
    public comment_base
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
    public weight_base,
    public group_base,
    public comment_base
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


template <
    class instance_tmpl,
    class attributes_quark_tmpl,
    class label_quark_tmpl
>
class attribute_data_base :
    public multi_data_base<instance_tmpl, attributes_quark_tmpl, label_quark_tmpl>
{
public:
    attribute_data_base()
    {
    }

    virtual ~attribute_data_base()
    {
    }
};



typedef sparse_vector_base<int, double> sparse_attributes;

typedef binary_instance_base<sparse_attributes> binstance;
typedef binary_data_base<binstance, quark> bdata;

typedef multi_candidate_base<sparse_attributes, int> mcandidate;
typedef multi_instance_base<mcandidate> minstance;
typedef multi_data_base<minstance, quark, quark> mdata;

typedef attribute_instance_base<sparse_attributes, int, data_traits> ainstance;
typedef attribute_data_base<ainstance, quark, quark> adata;

};

#endif/*__CLASSIAS_BASE_H__*/
