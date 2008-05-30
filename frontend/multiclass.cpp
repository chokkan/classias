/*
 *		Multi-class classification.
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
    // Split the line with tab characters.
    tokenizer field(line, '\t');
    if (!field.next()) {
        throw invalid_data("no field in the line", lines);
    }

    // Make sure that the first token (label) is not empty.
    if (field->empty()) {
        throw invalid_data("an empty label found", lines);
    }

    // Set the correct label.
    instance.set_label(labels(*field));

    // Loop over attributes.
    while (field.next()) {
        if (!field->empty()) {
            double value;
            std::string name;
            get_name_value(*field, name, value);
            instance.append(attrs(name), value);
        }
    }
}

template <
    class data_type,
    class attribute_quark_type,
    class label_quark_type
>
static void
read_stream(
    std::istream& is,
    data_type& data,
    attribute_quark_type& attrs,
    label_quark_type& labels,
    int group = 0
    )
{
    int lines = 0;
    typedef typename data_type::value_type instance_type;

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

        // Construct an instance object inside of the vector for efficiency.
        data.resize(data.size()+1);
        instance_type& inst = data.back();

        // Initialize the instance object.
        inst.set_group(group);
        read_line(line, inst, attrs, labels, lines);
    }
}

template <
    class training_data_type,
    class features_type,
    class attribute_quark_type,
    class label_quark_type,
    class raw_data_iterator_type
>
static void
convert_to_ranking(
    training_data_type& ddata,
    features_type& features,
    attribute_quark_type& attrs,
    label_quark_type& labels,
    raw_data_iterator_type begin,
    raw_data_iterator_type end
    )
{
    classias::classification_to_ranking(ddata, features, attrs, labels, begin, end);
}

template <
    class features_type,
    class value_type,
    class attribute_quark_type,
    class label_quark_type
>
static void
output_model(
    features_type& features,
    const value_type* weights,
    attribute_quark_type& attrs,
    label_quark_type& labels,
    const option& opt
    )
{
    std::ofstream ofs(opt.model.c_str());
    classias::output_model(ofs, features, weights, attrs, labels);
}

int multiclass_train(option& opt)
{
    return train_al<classias::cinstance>(opt);
}
