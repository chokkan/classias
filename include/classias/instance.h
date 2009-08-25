/*
 *		Classias instances.
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

#ifndef __CLASSIAS_INSTANCE_H__
#define __CLASSIAS_INSTANCE_H__

#include <vector>



namespace classias
{

/**
 * Binary instance.
 *
 *  This class implements an instance for binary classification.
 *
 *  @param  features_tmpl   The type of feature (=attribute) vector.
 *  @param  weight_base     The base class implementing instance weighting.
 *  @param  group_base      The base class managing group numbers.
 */
template <
    class features_tmpl
>
class binary_instance_base :
    public features_tmpl,
    public weight_base,
    public group_base
{
public:
    /// The type of a feature vector.
    typedef features_tmpl features_type;
    /// The type of an attribute identifier.
    typedef typename features_type::identifier_type attribute_type;

protected:
    /// The label (truth) of this instance.
    bool m_label;

public:
    /**
     * Constructs an object.
     */
    binary_instance_base() : m_label(false)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~binary_instance_base()
    {
    }

    /**
     * Sets the label.
     *  @param  l           The label.
     */
    inline void set_label(bool l)
    {
        m_label = l;
    }

    /**
     * Gets the label.
     *  @return bool        The label of this instance.
     */
    inline bool get_label() const
    {
        return m_label;
    }
};



/**
 * Multi-class instance.
 *
 *  This class implements an instance for multi-class classification.
 *
 *  @param  attributes_tmpl The type of attribute vector.
 *  @param  weight_base     The base class implementing instance weighting.
 *  @param  group_base      The base class managing group numbers.
 */
template <
    class attributes_tmpl
>
class multi_instance_base :
    public attributes_tmpl,
    public group_base,
    public weight_base
{
public:
    /// The type of an attribute vector.
    typedef attributes_tmpl attributes_type;
    /// The type of an attribute identifier.
    typedef typename attributes_type::identifier_type attribute_type;

protected:
    /// The label of this instance.
    int m_label;

public:
    /**
     * Constructs an object.
     */
    multi_instance_base() : m_label(-1)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~multi_instance_base()
    {
    }

    /**
     * Sets the label.
     *  @param  l           The label.
     */
    inline void set_label(int l)
    {
        m_label = l;
    }

    /**
     * Gets the label.
     *  @return bool        The label of this instance.
     */
    inline int get_label() const
    {
        return m_label;
    }

    /**
     * Returns the number of possible labels that can be assigned.
     *  @param  L           The total number of labels in the dataset.
     *  @return int         The number of possible labels for this instance.
     *                      The return value is always identical to L for
     *                      multi-class instances.
     */
    inline int num_labels(const int L) const
    {
        return L;
    }

    /**
     * Returns a read-only access to the attribute vector.
     *  @param  l           Reserved only for the compatibility with
     *                      candidate_instance_base class.
     *  @return const attributes_type&  The reference to the attribute vector
     *                                  associated with this instance.
     */
    inline const attributes_type& attributes(int l) const
    {
        return *this;
    }

    /**
     * Returns an access to the attribute vector.
     *  @param  l           Reserved only for the compatibility with
     *                      candidate_instance_base class.
     *  @return attributes_type&        The reference to the attribute vector
     *                                  associated with this instance.
     */
    inline attributes_type& attributes(int l)
    {
        return *this;
    }
};



/**
 * Candidate instance.
 *
 *  This class implements an instance for candidate classification.
 *  @param  candidate_tmpl  The type of a candidate.
 */
template <
    class attributes_tmpl
>
class candidate_instance_base :
    public weight_base,
    public group_base

{
public:
    /// A type representing an attribute vector.
    typedef attributes_tmpl attributes_type;
    /// A type representing a candidate (equivalent to attributes_type).
    typedef attributes_type candidate_type;
    /// A type providing a container of all candidates.
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
    /// The label of this instance.
    int m_label;

public:
    /**
     * Constructs an object.
     */
    candidate_instance_base() : m_label(-1)
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~candidate_instance_base()
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
     *  @retval int     The number of candidates associated with the instance.
     */
    inline size_type size() const
    {
        return candidates.size();
    }

    /**
     * Returns a read/write reference to a candidate.
     *  @param  i               The index number for a candidate.
     *  @retval candidate_type& Reference to the candidate.
     */
    inline candidate_type& operator[](size_type i)
    {
        return candidates[i];
    }

    /**
     * Returns a read-only reference to a candidate.
     *  @param  i                       The index number for a candidate.
     *  @retval const candidate_type&   Reference to the candidate.
     */
    inline const candidate_type& operator[](size_type i) const
    {
        return candidates[i];
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
     * Creates a new candidate.
     *  @retval candidate_type& The reference to the new candidate.
     */
    inline candidate_type& new_element()
    {
        candidates.push_back(candidate_type());
        return candidates.back();
    }

    /**
     * Sets the label.
     *  @param  l           The label.
     */
    inline void set_label(int l)
    {
        m_label = l;
    }

    /**
     * Gets the label.
     *  @return bool        The label of this instance.
     */
    inline int get_label() const
    {
        return m_label;
    }

    /**
     * Returns the number of candidates associated with the instance.
     *  @param  L           Ignored. Reserved only for the compatibility with
     *                      multi_instance_base class.
     *  @return int         The number of candidates associated with this
     *                      instance.
     */
    inline int num_labels(const int L) const
    {
        return this->size();
    }

    /**
     * Returns a read-only access to the attribute vector of a candidate.
     *  @param  l           The candidate label (index).
     *  @return const attributes_type&  The reference to the attribute vector
     *                                  associated for the candidate #l.
     */
    inline const attributes_type& attributes(int l) const
    {
        return this->operator[](l);
    }

    /**
     * Returns an access to the attribute vector of a candidate.
     *  @param  l           Reserved only for the compatibility with
     *                      candidate_instance_base class.
     *  @return attributes_type&        The reference to the attribute vector
     *                                  associated for the candidate #l.
     */
    inline attributes_type& attributes(int l)
    {
        return this->operator[](l);
    }
};


};


#endif/*__CLASSIAS_INSTANCE_H__*/
