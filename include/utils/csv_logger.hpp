/**
 * @file    csv_logger.hpp
 * @brief   Simple CSV logger for benchmark results.
 * @author  Kamil J.
 * @date    2025-07-10
 */

#pragma once

#include <fstream>
#include <string>
#include <vector>

/**
 * @class   CsvLogger
 * @brief   Minimal CSV logger utility.
 */
class CsvLogger
{
   public:
    /// Opens file (truncates if exists).
    CsvLogger(const std::string& filename, const std::vector<std::string>& headers) : out_(filename, std::ios::trunc)
    {
        write_row(headers);
    }

    /// Write a row (vector of columns as string).
    void write_row(const std::vector<std::string>& row)
    {
        for (size_t i = 0; i < row.size(); ++i)
        {
            if (i) out_ << ",";
            out_ << row[i];
        }
        out_ << "\n";
    }

    /// Ensure flush on destruction
    ~CsvLogger()
    {
        out_.flush();
    }

   private:
    std::ofstream out_;
};