/*
 *		A sample program for a binary classifier.
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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <classias/classias.h>
#include <classias/evaluation.h>
#include <classias/classify/linear/binary.h>
#include "strsplit.h"   // necessary for strsplit() and get_id_value().

typedef classias::auto_extensible_vector<double> model_type;
typedef classias::classify::linear_binary<model_type> classifier_type;

template <class model_type>
bool read_model(model_type& model, const char *fname)
{
    std::ifstream ifs(fname);
    if (ifs.fail()) {
        return false;
    }

    for (;;) {
        std::string line;
        std::getline(ifs, line);
        if (ifs.eof()) {
            break;
        }
        
        int pos = line.find('\t');
        if (pos != line.npos) {
            int fid = std::atoi(line.c_str());
            double weight = std::atof(line.c_str() + pos+1);
            model[fid] = weight;
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    std::istream& is = std::cin;
    std::ostream& os = std::cout;
    std::ostream& es = std::cerr;

    if (argc < 2) {
        es << "USAGE: " << argv[0] << " MODEL" << std::endl;
        es << std::endl;
        return 0;
    }

    model_type model;
    if (!read_model(model, argv[1])) {
        es << "ERROR: failed to read the model" << std::endl;
    }

    classias::accuracy acc;
    classifier_type cla(model);
    for (;;) {
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }

        // Split the line into fields with space characters.
        std::vector<std::string> fields;
        strsplit(fields, line);

        // The line must have at least a label and a feature.
        if (fields.size() > 2) {
            bool label = (fields[0] != "-1");
            cla.clear();

            // Loop over the rest of fields.
            for (size_t i = 1;i < fields.size();++i) {
                // Split the field into a feature identifier and value.
                int fid;
                double value;
                get_id_value(fields[i], fid, value, ':');

                cla.set(fid, value);
            }

            bool rl = (fields[0] != "-1");
            bool ml = static_cast<bool>(cla);
            acc.set(rl == ml);
            os << (ml ? "+1" : "-1") << std::endl;
            
        } else {
            os << std::endl;
        }
    }

    acc.output(es);

    return 0;
}
