/*
 *		Data I/O for multi-candidate classification.
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

#ifdef  HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <iostream>
#include <string>

#include <classias/classias.h>
#include <classias/train/lbfgs/candidate.h>

#include "option.h"
#include "tokenize.h"
#include "train.h"

/* Automatic generation of bias features is not supported. */

/*
<line>          ::= <comment> | <boi> | <eoi> | <unreg> | <candidate> | <br>
<comment>       ::= "#" <string> <br>
<boi>           ::= "@boi" [ <weight> ] <br>
<eoi>           ::= "@eoi" <br>
<unregularize>  ::= "@unregularize" ("\t" <label>)+ <br>
<instance>      ::= <class> [ <label> ] ("\t" <feature>)+ <br>
<class>         ::= "T" | "+" | "F" | "-"
<label>         ::= <name>
<feature>       ::= <name> [ ":" <weight> ]
<name>          ::= <string>
<weight>        ::= <numeric>
<br>            ::= "\n"
*/

template <
    class instance_type,
    class features_quark_type,
    class label_quark_type
>
static void
read_line(
    const std::string& line,
    instance_type& instance,
    features_quark_type& features,
    label_quark_type& labels,
    const option& opt,
    int lines = 0
    )
{
    std::string name;
    typedef typename instance_type::candidate_type candidate_type;

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

    name = *itv;

    // Set the truth value for this candidate.
    bool truth = false;
    if (name.compare(0, 1, "+") == 0 || name.compare(0, 1, "T") == 0) {
        truth = true;
    } else if (name.compare(0, 1, "-") == 0 || name.compare(0, 1, "F") == 0) {
        truth = false;
    } else {
        throw invalid_data("a class label must begins with '+', 'T', '-', or 'F'", lines);
    }

    // Create a new candidate.
    candidate_type& cand = instance.new_element();
    cand.set_truth(truth);
    cand.set_label(labels(name));

    // Set featuress for the instance.
    for (++itv;itv != values.end();++itv) {
        if (!itv->empty()) {
            double value;
            get_name_value(*itv, name, value, opt.value_separator);
            if (opt.ignore_filter || REGEX_SEARCH(name, opt.filter)) {
                cand.append(features(name), value);
            }
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

        // Read features that should not be regularized.
        if (line.compare(0, 13, "@unregularize") == 0) {
            if (!data.empty()) {
                throw invalid_data("Declarative @unregularize must precede an instance", lines);
            }

            // Feature names for unregularization.
            tokenizer values(line, opt.token_separator);
            tokenizer::iterator itv = values.begin();
            for (++itv;itv != values.end();++itv) {
                // Reserve early feature identifiers.
                data.attributes(*itv);
            }

            // Set the start index of the user features.
            data.set_user_feature_start(data.attributes.size());

        } else if (line.compare(0, 4, "@boi") == 0) {
            double value;
            std::string name;
            get_name_value(line, name, value, opt.value_separator);

            if (name == "@boi") {
                // Start of a new instance.
                instance_type& inst = data.new_element();
                inst.set_group(group);
                inst.set_weight(value);
            }

        } else if (line == "@eoi") {

        } else {
            // A new candidate.
            read_line(line, data.back(), data.attributes, data.labels, opt, lines);
        }
    }
}

template <
    class data_type
>
static void
finalize_data(
    data_type& data,
    const option& opt
    )
{
    typedef classias::int_t int_t;

    // Set positive labels.
    for (int_t l = 0;l < data.num_labels();++l) {
        if (opt.negative_labels.find(data.labels.to_item(l)) == opt.negative_labels.end()) {
            data.append_positive_label(l);
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
    typedef classias::int_t int_t;

    // Open a model file for writing.
    std::ofstream os(opt.model.c_str());

    // Output a model type.
    os << "@model" << '\t' << "candidate" << std::endl;

    // Store the feature weights.
    for (int_t i = 0;i < (int_t)data.attributes.size();++i) {
        value_type w = weights[i];
        if (w != 0.) {
            os <<
                w << '\t' <<
                data.attributes.to_item(i) << std::endl;
        }
    }
}

int candidate_train(option& opt)
{
    if (opt.generate_bias) {
        // Automatic generation of bias features is not supported for candidate type.
        throw invalid_algorithm("Automatic generation of bias features is not supported for 'candidate' type.");
    }

    // Branches for training algorithms.
    if (opt.algorithm == "logress.lbfgs") {
        return train<
            classias::cdata,
            classias::trainer_lbfgs_candidate<classias::cdata, classias::real_t>
        >(opt);
    } else {
        throw invalid_algorithm(opt.algorithm);
    }
    return 0;
}

bool candidate_usage(option& opt)
{
    // Branches for training algorithms.
    if (opt.algorithm == "logress.lbfgs") {
        classias::trainer_lbfgs_candidate<classias::cdata, classias::real_t> tr;
        tr.params().help(opt.os);
        return true;
    }
    return false;
}
