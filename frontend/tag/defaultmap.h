#ifndef __DEFAULTMAP_H__
#define __DEFAULTMAP_H__

#include <map>

template <
   class key_tmpl, 
   class type_tmpl, 
   class traits_tmpl = std::less<key_tmpl>, 
   class allocator_tmpl = std::allocator<std::pair<const key_tmpl, type_tmpl> > 
>
class defaultmap :
    public std::map<key_tmpl, type_tmpl, traits_tmpl, allocator_tmpl>
{
public:
    typedef std::map<key_tmpl, type_tmpl, traits_tmpl, allocator_tmpl> base_type;
    typedef type_tmpl type_type;
    typedef allocator_tmpl allocator_type;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::const_reference const_reference;
    typedef typename base_type::const_reverse_iterator const_reverse_iterator;
    typedef typename base_type::difference_type difference_type;
    typedef typename base_type::iterator iterator;
    typedef typename base_type::key_compare key_compare;
    typedef typename base_type::key_type key_type;
    typedef typename base_type::mapped_type mapped_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::reverse_iterator reverse_iterator;
    typedef typename base_type::size_type size_type;

    typedef typename base_type::value_type pair_type;
    typedef type_type value_type;

public:
    type_type& operator[](const key_type& key)
    {
        iterator it = this->find(key);
        if (it != this->end()) {
            return it->second;
        } else {
            std::pair<iterator, bool> ret = this->insert(pair_type(key, 0));
            return (ret.first)->second;
        }
    }

    const type_type& operator[](const key_type& key) const
    {
        iterator it = this->find(key);
        return (it != this->end() ? it->second : 0);
    }
};

#endif/*__DEFAULTMAP_H__*/
