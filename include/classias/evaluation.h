/*
 *		Utilities for evaluation.
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

#ifndef __CLASSIAS_EVALUATION_H__
#define __CLASSIAS_EVALUATION_H__

#include <iomanip>
#include <vector>

namespace classias
{

/**
 * Accuracy counter.
 */
class accuracy
{
protected:
    int m_m;    ///< The number of matches.
    int m_n;    ///< The total number of instances.

public:
    /**
     * Constructs an object.
     */
    accuracy() : m_m(0), m_n(0)
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~accuracy()
    {
    }

    /**
     * Increments the number of correct/incorrect instances.
     *  @param  b           A correctness of an instance.
     */
    inline void set(bool b)
    {
        m_m += static_cast<int>(b);
        ++m_n;
    }

    /**
     * Gets the accuracy.
     *  @return double      The accuracy.
     */
    inline operator double() const
    {
        return (0 < m_n ? m_m / (double)m_n : 0.);
    }

    /**
     * Outputs the accuracy score.
     *  @param  os          The output stream.
     */
    void output(std::ostream& os) const
    {
        double acc = (0 < m_n ? m_m / (double)m_n : 0);
        os << "Accuracy: " <<
            std::fixed << std::setprecision(4) << static_cast<double>(*this) <<
            std::setprecision(6) << 
            " (" << m_m << "/" << m_n << ")" << std::endl;
        os.unsetf(std::ios::fixed);
    }
};

/**
 * Counter for precision, recall, and F1 scores
 */
class precall
{
protected:
    /// A counter for each label.
    struct label_stat
    {
        int num_match;
        int num_reference;
        int num_prediction;

        label_stat() :
            num_match(0), num_reference(0), num_prediction(0)
        {
        }
    };

    int m_n;                ///< The number of labels.
    label_stat* m_stat;     ///< The label-wise stats.

public:
    /**
     * Constructs an object.
     *  @param  N           The number of labels.
     */
    precall(int N) : m_n(N)
    {
        m_stat = new label_stat[N];
    }

    /**
     * Destructs an object.
     */
    virtual ~precall()
    {
        delete[] m_stat;
    }

    /**
     * Sets a pair of predicted and reference labels.
     *  @param  p           The predicted label.
     *  @param  r           The reference label.
     */
    void set(int p, int r)
    {
        m_stat[r].num_reference++;
        m_stat[p].num_prediction++;
        if (r == p) m_stat[p].num_match++;
    }

    /**
     * Outputs micro-average precision, recall, F1 scores.
     *  @param  os          The output stream.
     *  @param  pb          The iterator for the first element of the
     *                      positive labels.
     *  @param  pe          The iterator just beyond the last element
     *                      of the positive labels.
     */
    template <class positive_iterator_type>
    void output_micro(
        std::ostream& os,
        positive_iterator_type pb,
        positive_iterator_type pe
        ) const
    {
        int num_match = 0;
        int num_reference = 0;
        int num_prediction = 0;

        for (positive_iterator_type it = pb;it != pe;++it) {
            num_match += m_stat[*it].num_match;
            num_reference += m_stat[*it].num_reference;
            num_prediction += m_stat[*it].num_prediction;
        }

        double precision = divide(num_match, num_prediction);
        double recall = divide(num_match, num_reference);
        double f1score = divide(2 * precision * recall, precision + recall);

        os << "Micro P, R, F1: " <<
            std::fixed << std::setprecision(4) << precision <<
            std::setprecision(6) << 
            " (" << num_match << "/" << num_prediction << ")" << ", " <<
            std::fixed << std::setprecision(4) << recall <<
            std::setprecision(6) <<
            " (" << num_match << "/" << num_reference << ")" << ", " <<
            std::fixed << std::setprecision(4) << f1score << std::endl;
        os << std::setprecision(6);
        os.unsetf(std::ios::fixed);
    }

    /**
     * Outputs macro-average precision, recall, F1 scores.
     *  @param  os          The output stream.
     *  @param  pb          The iterator for the first element of the
     *                      positive labels.
     *  @param  pe          The iterator just beyond the last element
     *                      of the positive labels.
     */
    template <class positive_iterator_type>
    void output_macro(
        std::ostream& os,
        positive_iterator_type pb,
        positive_iterator_type pe
        ) const
    {
        int n = 0;
        double precision = 0., recall = 0., f1 = 0.;

        for (positive_iterator_type it = pb;it != pe;++it) {
            double p = divide(m_stat[*it].num_match, m_stat[*it].num_prediction);
            double r = divide(m_stat[*it].num_match, m_stat[*it].num_reference);
            double f = divide(2 * p * r, p + r);
            precision += p;
            recall += r;
            f1 += f;
            ++n;
        }

        precision /= n;
        recall /= n;
        f1 /= n;

        os << "Macro P, R, F1: " << 
            std::fixed << std::setprecision(4) << precision << ", " <<
            std::fixed << std::setprecision(4) << recall << ", " <<
            std::fixed << std::setprecision(4) << f1 << std::endl;
        os << std::setprecision(6);
        os.unsetf(std::ios::fixed);
    }

protected:
    template <typename value_type>
    static inline double divide(value_type a, value_type b)
    {
        return (b != 0) ? (a / (double)b) : 0.;
    }
};

};

#endif/*__CLASSIAS_EVALUATION_H__*/
