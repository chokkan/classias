/*
 *      A simple tokenizer.
 *  
 *      Copyright (c) 2008 by Naoaki Okazaki
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions (known as zlib license):
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * Naoaki Okazaki <okazaki at chokkan dot org>
 *
 */

/* $Id$ */

#ifndef __TOKENIZE_H__
#define __TOKENIZE_H__

#include <string>

template <class char_type>
class basic_tokenizer
{
protected:
    typedef basic_tokenizer<char_type> tokenizer_type;
    typedef typename std::basic_string<char_type> string_type;

public:
    /**
     * Iterator class for tokenizer.
     */
    class iterator
    {
    protected:
        typedef typename string_type::const_iterator string_const_iterator;

        char_type m_sep;
        string_const_iterator m_it;
        string_const_iterator m_prev;
        string_const_iterator m_end;

        string_type m_token;

    public:
        /**
         * Constructs an iterator.
         */
        iterator()
            : m_sep(' ')
        {
        }

        /**
         * Constructs an iterator.
         *  @param  it          The iterator pointing to the first character.
         *  @param  end         The iterator pointing just beyond the last character.
         *  @param  sep         A separator.
         */
        iterator(
            string_const_iterator it,
            string_const_iterator end,
            char_type sep
            )
            : m_it(it), m_prev(it), m_end(end), m_sep(sep)
        {
            next();
        }

        /**
         * Destructs the iterator.
         */
        virtual ~iterator()
        {
        }

        /**
         * Constructs an iterator by cloning another iterator object.
         *  @param  x           The iterator that is to be cloned to the
         *                      target iterator.
         */
        iterator(const iterator& x)
        {
            operator=(x);
        }

        /**
         * Copies the content from another iterator object.
         *  @param  x           The iterator that is to be cloned to the
         *                      target iterator.
         *  @retval iterator&   The reference to this object.
         */
        inline iterator& operator=(const iterator& x)
        {
            m_it = x.m_it;
            m_prev = x.m_prev;
            m_end = x.m_end;
            m_sep = x.m_sep;
            m_token = x.m_token;
            return *this;
        }

        /**
         * Accesses to the current token.
         *  @retval string_type The current token.
         */
        inline const string_type& operator*() const
        {
            return m_token;
        }

        /**
         * Accesses to the pointer to the current token.
         *  @retval string_type*    The pointer to the current token.
         */
        inline const string_type* operator->() const
        {
            return &m_token;
        }

        /**
         * Advances to the next token.
         *  @retval iterator&   The reference to this object.
         */
        inline iterator& operator++()
        {
            next();
            return *this;
        }

        /**
         * Tests the iterator for equality with a specified iterator.
         *  @param  x           The iterator that is to be compared to the
         *                      target iterator for equality.
         *  @retval bool        \c true if the iterators are the same;
         *                      \c false if they are different.
         */
        inline bool operator==(const iterator& x) const
        {
            return (m_prev == x.m_prev);
        }

        /**
         * Tests the iterator for inequality with a specified iterator.
         *  @param  x           The iterator that is to be compared to the
         *                      target iterator for inequality.
         *  @retval bool        \c true if the iterators are different;
         *                      \c false if they are the same.
         */
        inline bool operator!=(const iterator& x) const
        {
            return !operator==(x);
        }

    protected:
        inline void next()
        {
            m_prev = m_it;

            if (m_it != m_end) {
                m_token.clear();
                for (;m_it != m_end;++m_it) {
                    if (*m_it == m_sep) {
                        ++m_it;
                        break;
                    }
                    m_token += *m_it;
                }
            }
        }
    };

protected:
    string_type m_str;
    char_type m_sep;
    iterator m_end;

public:
    /**
     * Constructs a tokenizer object.
     *  @param  str         the string to be tokenized.
     *  @param  sep         a separator character for tokenization.
     */
    basic_tokenizer(const string_type& str, const char_type sep = '\t')
        : m_str(str), m_sep(sep)
    {
        m_end = iterator(m_str.end(), m_str.end(), m_sep);
    }

    /**
     * Destructs the tokenizer object. 
     */
    virtual ~basic_tokenizer()
    {
    }

    /**
     * Returns a forward input iterator to the first token.
     *  @retval iterator        A forward input iterator (for read-only)
     *                          addressing the first token in the string or
     *                          to the location succeeding an empty token.
     */
    inline iterator begin() const
    {
        return iterator(m_str.begin(), m_str.end(), m_sep);
    }

    /**
     * Returns a forward input iterator pointing just beyond the last token.
     *  @retval iterator        A forward input iterator (for read-only)
     *                          addressing the end of the token.
     */
    inline iterator end() const
    {
        return m_end;
    }
};

typedef basic_tokenizer<char> tokenizer;

#endif/*__TOKENIZE_H__*/
