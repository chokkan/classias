/*
 *		Classias basic types.
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

#ifndef __CLASSIAS_TYPES_H__
#define __CLASSIAS_TYPES_H__

#include <vector>



namespace classias
{

/// Integer type.
typedef int int_t;
/// Float type.
typedef double real_t;

/**
 * Instance weight class.
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
};

typedef std::vector<double> weight_vector;

};

#endif/*__CLASSIAS_TYPES_H__*/
