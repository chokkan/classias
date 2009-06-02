/*
 *		Data I/O for attribute-based classification.
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

#include <iostream>
#include <string>

#include <classias/classias.h>
#include <classias/train/lbfgs/multi.h>

#include "option.h"
#include "tokenize.h"
#include "train.h"

/*
<line>          ::= <comment> | <instance> | <br>
<comment>       ::= "#" <string> <br>
<instance>      ::= <class> ("\t" <attribute>)+ <br>
<class>         ::= <string>
<attribute>     ::= <name> [ ":" <weight> ]
<name>          ::= <string>
<weight>        ::= <numeric>
<br>            ::= "\n"
*/

template <
    class instance_type,
    class attributes_quark_type,
    class label_quark_type
>
static void
read_line(
    const std::string& line,
    instance_type& instance,
    attributes_quark_type& attributes,
    label_quark_type& labels,
    const option& opt,
    int lines = 0
    )
{
    double value;
    std::string name;

    // Split the line with tab characters.
    tokenizer values(line, opt.token_separator);
    tokenizer::iterator itv = values.begin();
    if (itv == values.end()) {
        throw invalid_data("no field found in the line", lines);
    }

    // Make sure that the first token (class) is not empty.
    if (itv->empty()) {
        throw invalid_data("an empty label found", lines);
    }

    // Parse the instance label.
    get_name_value(*itv, name, value, opt.value_separator);

    // Set the instance label and weight.
    instance.set_label(labels(name));
    instance.set_weight(value);

    // Set attributes for the instance.
    for (++itv;itv != values.end();++itv) {
        if (!itv->empty()) {
            double value;
            std::string name;
            get_name_value(*itv, name, value, opt.value_separator);
            instance.append(attributes(name), value);
        }
    }

    // Include a bias feature if necessary.
    if (opt.generate_bias) {
        instance.append(attributes("@bias"), 1.);
    }
}

template <
    class data_type
>
static void
read_stream(
    std::istream& is,
    data_type& data,
    const option& opt,
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

        // Create a new instance.
        instance_type& inst = data.new_element();
        inst.set_group(group);

        read_line(line, inst, data.attributes, data.labels, opt, lines);
    }

    // Generate features.
    data.finalize();
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
    typedef typename classias::int_t int_t;
    typedef typename data_type::attributes_quark_type attributes_quark_type;
    typedef typename attributes_quark_type::item_type attribute_type;
    typedef typename data_type::labels_quark_type labels_quark_type;
    typedef typename labels_quark_type::item_type label_type;

    // Open a model file for writing.
    std::ofstream os(opt.model.c_str());

    // Output a model type.
    os << "@model" << '\t' << "attribute-label" << std::endl;

    // Output a set of labels.
    os << "@labels";
    for (int_t l = 0;l < data.num_labels();++l) {
        os << '\t' << data.labels.to_item(l);
    }
    os << std::endl;

    // Store the feature weights.
    for (int_t i = 0;i < data.num_features();++i) {
        value_type w = weights[i];
        if (w != 0.) {
            int_t a, l;
            data.feature_generator.backward(i, a, l);
            os << w << '\t'
                << data.attributes.to_item(a) << '\t'
                << data.labels.to_item(l) << std::endl;
        }
    }
}

int multi_train(option& opt)
{
    // Branches for training algorithms.
    if (opt.algorithm == "logress.lbfgs") {
        if (opt.type == option::TYPE_MULTI_SPARSE) {
            return train<
                classias::mdata,
                classias::trainer_lbfgs_multi<classias::mdata, double>
            >(opt);
        } else if (opt.type == option::TYPE_MULTI_DENSE) {
            return train<
                classias::ndata,
                classias::trainer_lbfgs_multi<classias::ndata, double>
            >(opt);
        }
    }
    throw invalid_algorithm(opt.algorithm);
}

bool multi_usage(option& opt)
{
    if (opt.algorithm == "logress.lbfgs") {
        classias::trainer_lbfgs_multi<classias::mdata, double> tr;
        tr.params().help(opt.os);
        return true;
    }
    return false;
}
