/*
 *		Ranker.
 *
 * Copyright (c) 2008, Naoaki Okazaki
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

/* $Id$ */

#ifdef  HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
#include <iterator>

#include <classias/base.h>

#include "option.h"
#include "tokenize.h"
#include "util.h"
#include "train.h"

template <
    class instance_type,
    class attribute_quark_type,
    class label_quark_type
>
static void
read_line(
    const std::string& line,
    instance_type& instance,
    attribute_quark_type& attrs,
    label_quark_type& labels,
    int lines = 0
    )
{
    typedef typename instance_type::candidate_type candidate_type;

    // Split the line with tab characters.
    tokenizer field(line, '\t');
    if (!field.next()) {
        throw invalid_data("no field found in the line", lines);
    }

    // Make sure that the first token (class) is not empty.
    if (field->empty()) {
        throw invalid_data("an empty label found", lines);
    }

    // Extract the label in the first token if any.
    std::string label;
    std::string::size_type pos = field->find(' ');
    if (pos != field->npos) {
        label = std::string(*field, pos+1);
    }

    // Set the binary class.
    bool torf = ((*field)[0] != '-');

    // Set the label.
    if (label.empty()) {
        label = *field;
    }

    // Create a new candidate.
    candidate_type& cand = instance.new_element();
    cand.torf = torf;
    cand.label = labels(label);

    // Set attributes for the instance.
    while (field.next()) {
        if (!field->empty()) {
            double value;
            std::string name;
            get_name_value(*field, name, value);
            cand.append(attrs(name), value);
        }
    }
}

template <
    class data_type
>
static void
read_stream(
    std::istream& is,
    data_type& data,
    int group = 0
    )
{
    int lines = 0;
    typedef typename data_type::instance_type instance_type;

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

        // Skip a comment line.
        if (line.compare(0, 1, "#") == 0) {
            continue;
        }

        if (line.compare(0, 3, "BOI") == 0) {
            // Start of a new instance.
            data.new_element();
        } else if (line.compare(0, 3, "EOI") == 0) {
            // End of a new instance.
        } else {
            // A new candidate.
            read_line(line, data.back(), data.attributes, data.labels, lines);
        }
    }
}

template <
    class data_type,
    class value_type
>
static void
output_model(
    data_type& data,
    const value_type* weights,
    const option& opt
    )
{
    typedef typename data_type::attribute_quark_type attribute_quark_type;
    typedef typename attribute_quark_type::value_type attribute_type;
    const attribute_quark_type& attributes = data.attributes;

    // Open a model file for writing.
    std::ofstream os(opt.model.c_str());

    for (attribute_type i = 0;i < attributes.size();++i) {
        value_type w = weights[i];
        if (w != 0.) {
            os << w << '\t' << attributes.to_item(i) << std::endl;
        }
    }
}

int ranker_train(option& opt)
{
    return train<classias::srdata>(opt);
}
