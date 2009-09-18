/*
 *		A string splitter to extract values expressed in a CSV-like format.
 *
 *		Copyright (c) 2009 Naoaki Okazaki
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

#ifndef	__STRSPLIT_H__
#define	__STRSPLIT_H__

#include <cstdlib>
#include <string>

template <class _Cont, class _Elem>
static int strsplit(
	_Cont& values,
	const std::basic_string<_Elem>& line,
	const _Elem sep = (const _Elem)' '
	)
{
	int ret = 0;

	// Initialize the container.
	values.clear();
	typename std::basic_string<_Elem>::const_iterator it = line.begin();
	while (it != line.end()) {
		std::basic_string<_Elem> value;
		for (;it != line.end();++it) {
			if (*it == sep) {
				++it;
				break;
			}
			value += *it;
		}
		values.push_back(value);
	}
	return ret;
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
