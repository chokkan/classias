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

#include <iostream>
#include <string>

template <class char_type>
class basic_tokenizer
{
protected:
    typedef basic_tokenizer<char_type> this_class;
    typedef typename std::basic_string<char_type> string_type;
    typedef typename string_type::const_iterator iterator_type;

    char_type sep;
    bool inl;
    string_type token;

    const string_type& line;
    iterator_type it;

public:
    basic_tokenizer(const string_type& _line, const char_type _sep = '\t') : inl(true), line(_line), sep(_sep)
    {
        it = line.begin();
    }

    virtual ~basic_tokenizer()
    {
    }

    operator bool() const
    {
        return inl;
    }

    const string_type& operator*() const
    {
        return token;
    }

    const string_type* operator->() const
    {
        return &token;
    }

    this_class& next()
    {
        if (it != line.end()) {
            token.clear();
            for (;it != line.end();++it) {
                if (*it == sep) {
                    ++it;
                    break;
                }
                token += *it;
            }
        } else {
            inl = false;
        }

        return *this;
    }
};

typedef basic_tokenizer<char> tokenizer;

template <class handler_base>
class basic_parser :
    public handler_base
{
public:
    basic_parser()
    {
    }

    virtual ~basic_parser()
    {
    }

    int parse(std::istream& is)
    {
        int lines = 0;

        for (;;) {
            // Read a line.
            std::string line;
            std::getline(is, line);
            if (is.eof()) {
                break;
            }
            ++lines;

            // Skip an empty line.
            if (line.empty()) {
                continue;
            }

            // Find a comment line.
            if (line.compare(0, 1, "#") == 0) {
                handler_base::comment(line, lines);
                continue;
            }

            // Notify a start of tokens.
            handler_base::start_tokens(line, lines);

            // Split the line with tab characters.
            tokenizer field(line, '\t');
            while (field.next()) {
                handler_base::token(*field, line);
            }

            // Notify an end of tokens.
            handler_base::end_tokens(line, lines);
        }
    }
};

#endif/*__TOKENIZE_H__*/
