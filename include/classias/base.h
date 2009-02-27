#ifndef __CLASSIAS_BASE_H__
#define __CLASSIAS_BASE_H__

#include <string>

namespace classias
{

/**
 * Truth class.
 *
 *  This class implements the base interface indicating whether an inherited
 *  object is a true instance/candidate.
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
     * Constructs an object that is a copy of another object.
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
     * Constructs an object that is a copy of another object.
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



/**
 * Feature weight class.
 *
 *  This class implements the base interface to represent a weight of an
 *  instance.
 */
class weight_base
{
public:
    /// The type representing the weight.
    typedef double weight_type;

protected:
    /// The instance weight.
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
     *  @param  weight      The weight value used to initialize the object.
     */
    weight_base(const weight_type& weight) : m_weight(weight)
    {
    }

    /**
     * Constructs an object that is a copy of another object.
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
     * Assigns a new weight to the object.
     *  @param  rho             The source object.
     *  @retval weight_type&    The reference to this object.
     */
    weight_base& operator=(const weight_base& rho)
    {
        m_weight = rho.m_weight;
        return *this;
    }

    /**
     * Tests the equality of two weight objects.
     *  @param  x           A weight object.
     *  @param  y           Another weight object.
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
     * Tests the inequality of two weight objects.
     *  @param  x           A weight object.
     *  @param  y           Another weight object.
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
     * Assigns a new weight to the object.
     *  @param  weight      The weight value.
     */
    inline void set_weight(weight_type weight)
    {
        m_weight = weight;
    }

    /**
     * Obtains the current weight value.
     *  @retval weight_type The current value.
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
     * Constructs an object that is a copy of another object.
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



/**
 * Comment class.
 *
 * The class implements the base interface for instances with comments.
 */
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
     * Constructs an object that is a copy of another object.
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

};

#endif/*__CLASSIAS_BASE_H__*/
