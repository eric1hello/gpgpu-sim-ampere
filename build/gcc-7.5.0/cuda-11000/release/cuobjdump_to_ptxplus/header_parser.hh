/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_HEADER_MEDIA_ERLI_3BE200E4_F4A3_4EC6_9E05_B18F1D597DF2_PROJ_SOC_SOC_GPGPU_SIM_ZHAOSIYING_ACCEL_SIM_GPGPU_SIM_DISTRIBUTION_BUILD_GCC_7_5_0_CUDA_11000_RELEASE_CUOBJDUMP_TO_PTXPLUS_HEADER_PARSER_HH_INCLUDED
# define YY_HEADER_MEDIA_ERLI_3BE200E4_F4A3_4EC6_9E05_B18F1D597DF2_PROJ_SOC_SOC_GPGPU_SIM_ZHAOSIYING_ACCEL_SIM_GPGPU_SIM_DISTRIBUTION_BUILD_GCC_7_5_0_CUDA_11000_RELEASE_CUOBJDUMP_TO_PTXPLUS_HEADER_PARSER_HH_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int header_debug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    DOTVERSION = 258,
    DOTTARGET = 259,
    DOTENTRY = 260,
    DOTPARAM = 261,
    DOTU64 = 262,
    DOTU32 = 263,
    DOTU16 = 264,
    DOTB32 = 265,
    DOTF32 = 266,
    IDENTIFER = 267,
    DECLITERAL = 268,
    LEFTPAREN = 269,
    RIGHTPAREN = 270
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 41 "header.y" /* yacc.c:1909  */

  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;

#line 78 "/media/erli/3be200e4-f4a3-4ec6-9e05-b18f1d597df2/proj_soc/soc/gpgpu-sim_zhaosiying/accel-sim/gpgpu-sim_distribution/build/gcc-7.5.0/cuda-11000/release/cuobjdump_to_ptxplus/header_parser.hh" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE header_lval;

int header_parse (void);

#endif /* !YY_HEADER_MEDIA_ERLI_3BE200E4_F4A3_4EC6_9E05_B18F1D597DF2_PROJ_SOC_SOC_GPGPU_SIM_ZHAOSIYING_ACCEL_SIM_GPGPU_SIM_DISTRIBUTION_BUILD_GCC_7_5_0_CUDA_11000_RELEASE_CUOBJDUMP_TO_PTXPLUS_HEADER_PARSER_HH_INCLUDED  */
