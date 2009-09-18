/*
 *		Utilities for splitting a string.
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

#ifndef	__STRSPLIT_H__
#define	__STRSPLIT_H__

#include <cstdlib>
#include <string>

template <class container_type, class char_type>
static void strsplit(
	container_type& values,
	const std::basic_string<char_type>& line,
	const char_type sep = (const char_type)' '
	)
{
	// Initialize the container.
	values.clear();

	typename std::basic_string<char_type>::const_iterator it = line.begin();
	while (it != line.end()) {
		std::basic_string<char_type> value;
		for (;it != line.end();++it) {
			if (*it == sep) {
				++it;
				break;
			}
			value += *it;
		}
		values.push_back(value);
	}
}

static void
get_id_value(
    const std::string& str, int& id, double& value, char separator)
{
    size_t col = str.rfind(separator);
    if (col == str.npos) {
        id = std::atoi(str.c_str());
        value = 1.;
    } else {
        id = std::atoi(str.substr(0, col).c_str());
        value = std::atof(str.c_str() + col + 1);
    }
}

#endif/*__STRSPLIT_H__*/
