/*
 *		Quark utilities for Classias.
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

#ifndef __CLASSIAS_QUARK_H__
#define __CLASSIAS_QUARK_H__

#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#if defined(HAVE_UNORDERED_MAP)
#include <unordered_map>
#define UNORDERED_MAP   std::tr1::unordered_map
namespace std {
    namespace tr1 {
        template<>
        struct hash<std::pair<int, int> > {
            size_t operator()(const std::pair<int, int>& item) const
            {
                return (size_t)((item.first << 8) + (item.second & 0xFF));
            }
        };
    };
};
#else
#include <map>
#define UNORDERED_MAP   std::map

#endif

#elif defined __GNUC__

#if defined(HAVE_BOOST_UNORDERED_MAP_HPP)
#include <boost/unordered_map.hpp>
#define UNORDERED_MAP   boost::unordered_map
namespace boost {
    template<>
    struct hash<std::pair<int, int> > {
        size_t operator()(const std::pair<int, int>& item) const
        {
            return (size_t)((item.first << 8) + (item.second & 0xFF));
        }
    };
};

#elif defined(HAVE_TR1_UNORDERED_MAP)
#include <tr1/unordered_map>
#define UNORDERED_MAP   std::tr1::unordered_map
namespace std {
    namespace tr1 {
        template<>
        struct hash<std::pair<int, int> > {
            size_t operator()(const std::pair<int, int>& item) const
            {
                return (size_t)((item.first << 8) + (item.second & 0xFF));
            }
        };
    };
};

#else
#include <map>
#define UNORDERED_MAP   std::map

#endif


#endif



namespace classias {

/**
 * Exception class for \ref quark_base and \ref quark2_base.
 */
class quark_error : public std::out_of_range
{
public:
    /**
     * Constructs an exception object.
     *  @param  msg         The error message.
     */
    explicit quark_error(const std::string& msg)
        : std::out_of_range(msg)
    {
    }
};



/**
 * Quark for associating an item with an identifier.
 *
 *  @param  item_base       The type of an item.
 */
template <class item_base>
class quark_base {
public:
    /// The type representing an item.
    typedef item_base item_type;
    /// The type of this class.
    typedef quark_base<item_base> this_class;

    /// The type implementing a vector of items.
    typedef std::vector<item_type> inverse_map_type;
    /// The type representing a unique identifier.
    typedef typename inverse_map_type::size_type value_type;
    /// The type associating an item to its unique identifier.
    typedef UNORDERED_MAP<item_type, value_type> forward_map_type;

protected:
    /// Forward mapping: (item0, item1) -> value.
    forward_map_type m_fwd;
    /// Inverse mapping: value -> (item0, item1).
    inverse_map_type m_inv;

public:
    /**
     * Constructs the object.
     */
    quark_base()
    {
    }

    /**
     * Constructs the object by copying the source object.
     *  @param  src             The source object.
     */
    quark_base(const this_class& src)
    {
        m_fwd = src.m_fwd;
        m_inv = src.m_inv;
    }

    /**
     * Destructs the object.
     */
    virtual ~quark_base()
    {
    }

    /**
     * Copies another object to this object.
     *  @param  src             The source object.
     *  @return this_class&     The reference to this object.
     */
    this_class& operator=(const this_class& src)
    {
        m_fwd = src.m_fwd;
        m_inv = src.m_inv;
        return *this;
    }

    /**
     * Returns the number of item-identifier associations.
     *  @retval value_type      The number of associations between items and
     *                          identifiers.
     */
    inline value_type size() const
    {
        return m_fwd.size();
    }

    /**
     * Tests whether an item has an identifier assigned.
     *  @param  x               The item.
     *  @retval bool            \c true if the item is known.
     */
    inline bool exists(const item_type& x)
    {
        return m_fwd.find(x) != m_fwd.end();
    }

    /**
     * Assigns the unique identifier for an item.
     *  If the item is unknown, this function assigns a new unique identifier
     *  to the item and return it.
     *  @param  x               The item.
     *  @return value_type      The unique identifier.
     */
    inline value_type operator() (const item_type& x)
    {
        return associate(x);
    }

    /**
     * Assigns a unique identifier for a new item.
     *  If the item is unknown, this function assigns a new unique identifier
     *  to the item and return it. If the item is known, this function returns
     *  the existing identifier that was associated with the item.
     *  @param  x               The item.
     *  @return value_type      The unique identifier.
     */
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

    /**
     * Returns the unique identifier for an item.
     *  If the item is unknown, this function throws quark_error.
     *  @param  x               The item.
     *  @return value_type      The unique identifier.
     *  @throws quark_error.
     */
    inline value_type to_value(const item_type& x) const
    {
        typename forward_map_type::const_iterator it = m_fwd.find(x);
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            throw quark_error("Unknown forward mapping");
        }           
    }

    /**
     * Returns the unique identifier for an item.
     *  If the item is unknown, this function returns the default identifier.
     *  @param  x               The item.
     *  @param  def             The default identifier if the item is unknown.
     *  @return value_type      The unique identifier.
     */
    inline value_type to_value(const item_type& x, const value_type& def) const
    {
        typename forward_map_type::const_iterator it = m_fwd.find(x);
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            return def;
        }           
    }

    /**
     * Returns the item for the unique identifier.
     *  If the unique identifier is unknown, this function throws quark_error.
     *  @param  v               The unique identifier.
     *  @return item_type&      The reference to the item associated with
     *                          the identifier.
     *  @throws quark_error.
     */
    inline const item_type& to_item(const value_type& v) const
    {
        if (v < m_inv.size()) {
            return m_inv[v];
        } else {
            throw quark_error("Unknown inverse mapping");
        }
    }
};



/**
 * Quark for associating a pair of items with an identifier.
 *
 *  @param  item0_base      The type of an item #0.
 *  @param  item1_base      The type of an item #1.
 */
template <class item0_base, class item1_base>
class quark2_base {
public:
    /// The type representing an item #0.
    typedef item0_base item0_type;
    /// The type representing an item #1.
    typedef item1_base item1_type;

    /// The type representing a pair of items.
    typedef std::pair<item0_type, item1_type> elem_type;
    /// The type implementing a vector of pairs of items.
    typedef std::vector<elem_type> inverse_map_type;
    /// The type representing a unique identifier.
    typedef typename inverse_map_type::size_type value_type;
    /// The type associating a pair of items to its unique identifier.
    typedef UNORDERED_MAP<elem_type, value_type> forward_map_type;

protected:
    /// Forward mapping: (item0, item1) -> value.
    forward_map_type m_fwd;
    /// Inverse mapping: value -> (item0, item1).
    inverse_map_type m_inv;

public:
    /**
     * Constructs the object.
     */
    quark2_base()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~quark2_base()
    {
    }

    /**
     * Returns the number of item-identifier associations.
     *  @retval value_type      The number of associations between items and
     *                          identifiers.
     */
    inline value_type size() const
    {
        return m_fwd.size();
    }

    /**
     * Tests whether a pair of items has an identifier assigned.
     *  @param  x               The item #0.
     *  @param  y               The item #1.
     *  @retval bool            \c true if the pair of items is known.
     */
    inline bool exists(const item0_type& x, const item1_base& y) const
    {
        return m_fwd.find(elem_type(x, y)) != m_fwd.end();
    }

    /**
     * Assigns the unique identifier for a pair of items.
     *  If the pair is unknown, this function assigns a new unique identifier
     *  to the pair and return it.  If the item is known, this function returns
     *  the existing identifier that was associated with the item.
     *  @param  x               The item #0.
     *  @param  y               The item #1.
     *  @return value_type      The unique identifier.
     */
    inline value_type operator() (const item0_type& x, const item1_base& y)
    {
        return associate(x, y);
    }

    /**
     * Assigns the unique identifier for a pair of items.
     *  If the pair is unknown, this function assigns a new unique identifier
     *  to the pair and return it.  If the item is known, this function returns
     *  the existing identifier that was associated with the item.
     *  @param  x               The item #0.
     *  @param  y               The item #1.
     *  @return value_type      The unique identifier.
     */
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

    /**
     * Returns the unique identifier for a pair of items.
     *  If the pair is unknown, this function throws quark_error.
     *  @param  x               The item #0.
     *  @param  y               The item #1.
     *  @return value_type      The unique identifier.
     *  @throws quark_error.
     */
    inline value_type to_value(const item0_type& x, const item1_type& y) const
    {
        typename forward_map_type::const_iterator it = m_fwd.find(elem_type(x, y));
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            throw quark_error("Unknown forward mapping");
        }           
    }

    /**
     * Returns the unique identifier for a pair of items.
     *  If the pair is unknown, this function returns the default identifier.
     *  @param  x               The item #0.
     *  @param  y               The item #1.
     *  @param  def             The default identifier if the pair is unknown.
     *  @return value_type      The unique identifier.
     */
    inline value_type to_value(const item0_type& x, const item1_type& y, const value_type& def) const
    {
        typename forward_map_type::const_iterator it = m_fwd.find(elem_type(x, y));
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            return def;
        }           
    }

    /**
     * Returns the pair for the unique identifier.
     *  If the unique identifier is unknown, this function throws quark_error.
     *  @param  v               The unique identifier.
     *  @param  x               The reference to item #0.
     *  @param  y               The reference to item #1.
     *  @throws quark_error.
     */
    inline void to_item(const value_type& v, item0_type& x, item1_base& y) const
    {
        if (v < m_inv.size()) {
            x = m_inv[v].first;
            y = m_inv[v].second;
        } else {
            throw quark_error("Unknown inverse mapping");
        }
    }
};

/// The string quark.
typedef quark_base<std::string> quark;

};

#endif/*__CLASSIAS_QUARK_H__*/
