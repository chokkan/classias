/*
 *		Instances for Classias.
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

/* $Id:$ */

#ifndef __CLASSIAS_INSTANCE_H__
#define __CLASSIAS_INSTANCE_H__

#include <map>
#include <utility>
#include <vector>

namespace classias
{



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
    label_type m_label;

    labeled_candidate_base()
        : m_instance(NULL), m_label(0)
    {
    }

    labeled_candidate_base(const instance_type* inst)
        : m_instance(inst), m_label(0)
    {
    }

    labeled_candidate_base(
        const instance_type* inst,
        const label_type& label
        )
        : m_instance(inst), m_label(label)
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
            int fid = m_instance->m_traits->forward(it->first, m_label);
            if (0 <= fid) {
                s += (double)v[fid] * (double)it->second;
            }
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
            int fid = m_instance->m_traits->forward(it->first, m_label);
            if (0 <= fid) {
                v[fid] += scale * (double)it->second;
            }
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
class candidate_base : 
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
    candidate_base()
    {
    }

    /**
     * Destructs a candidate.
     */
    virtual ~candidate_base()
    {
    }
};





/**
 * Ranking candidates.
 *
 *  This class implements a linear array of ranking candidates.
 */
template <class candidate_base>
class candidates_base
{
public:
    /// A type representing an features.
    typedef candidate_base candidate_type;
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
template <
    class features_tmpl,
    class data_traits_tmpl
>
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
    typedef data_traits_tmpl traits_type;

    /**
     * Constructs an object.
     */
    binary_instance_base()
    {
    }

    /**
     * Constructs an object.
     */
    binary_instance_base(traits_type* traits)
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
template <
    class candidate_tmpl,
    class data_traits_tmpl
>
class multi_instance_base :
    public candidates_base<candidate_tmpl>,
    public weight_base,
    public group_base,
    public comment_base
{
public:
    /// The type of a candidate.
    typedef candidate_tmpl candidate_type;

    typedef data_traits_tmpl traits_type;

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
     * Constructs an object.
     */
    multi_instance_base(traits_type* traits)
    {
    }

    /**
     * Destructs an object.
     */
    virtual ~multi_instance_base()
    {
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
    traits_type* m_traits;

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

    attribute_instance_base(traits_type* traits)
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

    void examine()
    {
        typename attributes_type::const_iterator it;
        for (it = attributes.begin();it != attributes.end();++it) {
            m_traits->examine(it->first, this->get_label());
        }
    }
};


};


#endif/*__CLASSIAS_INSTANCE_H__*/
