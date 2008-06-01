#ifndef __CLASSIAS_QUARK_H__
#define __CLASSIAS_QUARK_H__

#include <map>
#include <stdexcept>
#include <vector>

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
    quark_error(const std::string& msg)
        : std::out_of_range(msg)
    {
    }
};

template <class item_base>
class quark_base {
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
    /**
     * Constructs the object.
     */
    quark_base()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~quark_base()
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
     * Tests whether an item has an identifier assigned.
     *  @param  x               The item.
     *  @retval bool            \c true if the item 
     */
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
            throw quark_error("Unknown forward mapping");
        }           
    }

    inline const item_type& to_item(const value_type& v) const
    {
        if (v < m_inv.size()) {
            return m_inv[v];
        } else {
            throw quark_error("Unknown inverse mapping");
        }
    }
};



template <class item0_base, class item1_base>
class quark2_base {
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
    quark2_base()
    {
    }

    virtual ~quark2_base()
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
            throw quark_error("Unknown forward mapping");
        }           
    }

    inline value_type to_value(const item0_type& x, const item1_type& y, const value_type& def) const
    {
        typename forward_map_type::const_iterator it = m_fwd.find(elem_type(x, y));
        if (it != m_fwd.end()) {
            return it->second;
        } else {
            return def;
        }           
    }

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

typedef quark_base<std::string> quark;
typedef quark2_base<std::string, std::string> quark2;

};

#endif/*__CLASSIAS_QUARK_H__*/
