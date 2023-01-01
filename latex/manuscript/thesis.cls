\ProvidesClass{thesis}[2021/06/21 PhD Thesis]

\PassOptionsToClass{a4paper}{memoir}
\PassOptionsToClass{12pt}{memoir}
\PassOptionsToClass{oldfontcommands}{memoir}
\LoadClass{memoir}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Packages for fonts
\RequirePackage[LGR, OT1]{fontenc}
\renewcommand*\familydefault{\sfdefault}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern}

\DeclareMathAlphabet{\mathgtt}{LGR}{cmtt}{m}{n}

% Packages
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[backref=true, backend=biber, maxbibnames=10, maxcitenames=2, citestyle=authoryear-comp, bibstyle=authoryear, uniquelist=false, uniquename=false, dashed=false, natbib=true, sortcites=true, sorting=ynt]{biblatex}
\RequirePackage{csquotes}
\RequirePackage{minitoc}
\RequirePackage{bbold}
\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amsfonts}
\RequirePackage{mathtools}
\RequirePackage{alphabeta}
\RequirePackage{xspace}
\RequirePackage{booktabs}
\RequirePackage{geometry}
\RequirePackage{thm-restate}
\RequirePackage{thmtools}
\RequirePackage{stmaryrd}
\RequirePackage{framed}
\RequirePackage{mdframed}
\RequirePackage[many]{tcolorbox}
\RequirePackage[chapter]{algorithm}
\RequirePackage[noend]{algpseudocode}
\RequirePackage{appendix}
\RequirePackage[shortlabels]{enumitem}
\RequirePackage{float}
\RequirePackage[misc]{ifsym}
\RequirePackage{tikz}
\RequirePackage{etoolbox}
\RequirePackage{caption,subcaption}
\RequirePackage[subpreambles=true]{standalone}
\RequirePackage{cleveref}
\RequirePackage{fancyhdr}
\RequirePackage{arydshln}
\RequirePackage{wasysym}
\RequirePackage{multirow}
\RequirePackage{multicol}
\RequirePackage{longtable}
\RequirePackage{pdfpages}
\RequirePackage[]{pdflscape}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% memoir options

% Inspired from http://mirrors.ctan.org/macros/latex/contrib/memoir/memman.pdf
% (page 80)
\renewcommand{\partnamefont}{\normalfont\LARGE\scshape}
\renewcommand{\partnumfont}{\normalfont\LARGE\scshape}
\renewcommand{\parttitlefont}{\bfseries\HUGE\scshape}


% Inspired from the Pedersen's style in
% http://mirrors.ctan.org/macros/latex/contrib/memoir/memman.pdf (page 404)
\renewcommand*{\colorchapnum}{}
\renewcommand*{\colorchaptitle}{}
\makechapterstyle{pedersen}{%
\setlength{\beforechapskip}{-50pt}
\setlength{\afterchapskip}{70pt}
\renewcommand*{\chapnamefont}{\normalfont\LARGE\itshape}
\renewcommand*{\chapnumfont}{\normalfont\huge\scshape\colorchapnum}
\renewcommand*{\chaptitlefont}{\normalfont\HUGE\scshape\colorchaptitle}
\renewcommand*{\afterchapternum}{}
\renewcommand*{\printchaptername}{}
\setlength{\midchapskip}{20mm}
\renewcommand*{\chapternamenum}{}
\renewcommand*{\printchapternum}{%
\sidebar{\raisebox{0pt}[0pt][0pt]{\makebox[0pt][l]{%
\resizebox{!}{\midchapskip}{\chapnumfont\thechapter}}}}}
\renewcommand*{\printchaptertitle}[1]{\chaptitlefont ##1}}
\setsecnumdepth{subsubsection}
\chapterstyle{pedersen}

\setlength{\parindent}{0cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fancyhdr options

% Based on https://tex.stackexchange.com/questions/127151/dictionary-style-header-problem-with-fancyhdr
\newcommand{\mymarks}{
{\ifthenelse{\equal{}{\rightmark}}{\nouppercase{\leftmark}}{\nouppercase{\rightmark}}}}

\fancypagestyle{headings}{%
\fancyhf{}
\fancyfoot[C]{{\bf --~\thepage~--}}
\fancyhead[LE,RO]{}
\fancyhead[RE,LO]{\mymarks}
\renewcommand{\headrulewidth}{0.04cm}%
\renewcommand{\footrulewidth}{0cm}%
}

\fancypagestyle{plain}{
\fancyhf{}
\fancyfoot[C]{{\bf --~\thepage~--}}
\renewcommand{\headrulewidth}{0cm}
\renewcommand{\footrulewidth}{0cm}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hyperref options
% Based on https://yelangya3826850.github.io/2017/03/26/LateX-Beamer/
\pdfstringdefDisableCommands{
    \def\\{}
    \def\texttt#1{<#1>}
    \def\medskip{}
    \def\smallskip{}
    \def\vspace{}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% biblatex options
\addbibresource{publications.bib}
\addbibresource{references.bib}
\defbibfilter{references}{not keyword={journal} and not keyword={conference} and not keyword={workshop} and not keyword={nat-conference} and not keyword={report}}

\setlength{\bibhang}{0pt}
\setlength\bibitemsep{\baselineskip}

\renewbibmacro{in:}{}
\DeclareNameAlias{sortname}{given-family}
 
 % from \renewbibmacro*{date+extradate} in authoryear.bbx
 \renewbibmacro*{date+extradate}{}
  \renewbibmacro*{date}{%
    \iffieldundef{labelyear}
      {}
      {\printtext[parens]{%
         \iflabeldateisdate
           {\printfield{issue}%
            \setunit*{\addspace}%
            \printdateextra}
           {\printlabeldateextra}}}}
  
   % https://tex.stackexchange.com/questions/380299/change-position-of-year-field-in-authoryear-biblatex-style
  \renewbibmacro*{issue+date}{%
  \printtext[parens]{%
    \printfield{issue}%
    \setunit*{\addspace}%
     \printdate
     }%
  \newunit}
  
  % from \newbibmacro*{maintitle+booktitle} in standard.bbx
  \renewbibmacro*{maintitle+booktitle}{%
  \iffieldundef{booktitle}{}{\printfield{booktitle}}%
  \newunit}
  
  % from newbibmacro*{chapter+pages} in standard.bbx
  \renewbibmacro*{chapter+pages}{}
  % from newbibmacro*{note+pages} in standard.bbx  
  \renewbibmacro*{note+pages}{}
 
  % from \newbibmacro*{publisher+location+date} in standard.bbx
  \renewbibmacro*{publisher+location+date}{
  \printlist{publisher}
  \newunit
  \usebibmacro{date}
  \newunit}
  
  % from \newbibmacro*{journal+issuetitle} in standard.bbx
  \renewbibmacro*{journal+issuetitle}{%
  \usebibmacro{journal}%
  \newunit%
  \printfield{series}%
  \newunit
  \usebibmacro{volume+number+eid}%
  \newunit
  \usebibmacro{issue+date}%
  \newunit}

% https://github.com/plk/biblatex/issues/899
\DefineBibliographyStrings{english}{
  andothers = {\mkbibemph{et\addabbrvspace al\adddot}}
}

\renewcommand*{\mkbibnamefamily}{\textsc}
\renewcommand*{\mkbibnamegiven}{\textsc}
\renewcommand*{\mkbibnamesuffix}{\textsc}
\renewcommand*{\mkbibnameprefix}{\textsc}

\DeclareFieldFormat*{title}{#1}
\DeclareFieldFormat*{booktitle}{\mkbibemph{#1}}
\DeclareListFormat*{publisher}{\mkbibemph{#1}}
\DeclareFieldFormat*{series}{\mkbibemph{#1}}

% https://tex.stackexchange.com/questions/524911/how-to-change-the-format-of-backref-i-want-cited-on-page-x
\renewbibmacro*{pageref}{%
  \iflistundef{pageref}
    {}
    {\newline\rule[0.8mm]{1cm}{0.5mm}\textbf{\footnotesize
       \ifnumgreater{\value{pageref}}{1}
         {\bibstring{backrefpages}\xspace}
         {\bibstring{backrefpage}\xspace}
       \printlist[pageref][-\value{listtotal}]{pageref}}}}
\DefineBibliographyStrings{english}{
  backrefpage={Cited on page},
  backrefpages={Cited on pages}
}

% https://tex.stackexchange.com/questions/126226/how-do-i-instruct-fullcite-to-use-maxbibnames-rather-than-maxcitenames
\newcommand{\printpublication}[1]{\AtNextCite{\defcounter{maxnames}{99}}\fullcite{#1}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minitoc options
\setcounter{minitocdepth}{2}

\renewcommand{\mtcoffset}{-8pt} 
\setlength{\mtcindent}{-8pt}
\renewcommand{\mtifont}{\large\sffamily\bfseries}
\renewcommand{\mtcSSSfont}{\footnotesize}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xcolor options

% Paul Tol's  "Vibrant" color scheme
% https://personal.sron.nl/~pault/data/colourschemes.pdf
\definecolor{blue}{HTML}{0077BB}
\definecolor{cyan}{HTML}{33BBEE}
\definecolor{green}{HTML}{009988}
\definecolor{orange}{HTML}{EE7733}
\definecolor{red}{HTML}{CC3311}
\definecolor{magenta}{HTML}{EE3377}
\definecolor{grey}{HTML}{BBBBBB}

\definecolor{proofcolor}{HTML}{F5F5F5}
\colorlet{theoremcolor}{blue}

\newcommand\black[1]{\textcolor{black}{#1}}
\newcommand\blue[1]{\textcolor{blue}{#1}}
\newcommand\cyan[1]{\textcolor{cyan}{#1}}
\newcommand\green[1]{\textcolor{green}{#1}}
\newcommand\orange[1]{\textcolor{orange}{#1}}
\newcommand\red[1]{\textcolor{red}{#1}}
\newcommand\magenta[1]{\textcolor{magenta}{#1}}
\newcommand\grey[1]{\textcolor{grey}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% amsthm/thmtools/framed/mdframed options

\renewcommand{\qedsymbol}{$\blacksquare$}

% Based on https://tex.stackexchange.com/questions/484676/styling-amsthm-theorem-using-thmtools
\renewenvironment{leftbar}{%
  \def\FrameCommand{{\color{theoremcolor}\vrule width 3pt} \hspace{10pt}}%
  \MakeFramed {\advance\hsize-\width \FrameRestore}}%
 {\endMakeFramed}

 \declaretheoremstyle[
 preheadhook=\begin{leftbar},%
 postfoothook=\end{leftbar},%
 ]{theoremstyle}

% Based on https://tex.stackexchange.com/questions/34405/shaded-theorems-thmtools-spanning-pages
\declaretheoremstyle[
preheadhook={\begin{mdframed}[backgroundcolor=proofcolor, topline=false,bottomline=false,leftline=false,rightline=false]},%
postfoothook=\end{mdframed},%
headfont=\normalfont\itshape,%
qed=$\blacksquare$,%
headpunct={.},%
notebraces={}{},%
notefont=\normalfont\itshape,%
]{proofstyle}

\declaretheorem[name={\proofname}, style=proofstyle, unnumbered]{newproof}
\declaretheorem[style=theoremstyle, numberwithin=section]{definition}
\declaretheorem[style=theoremstyle, numberwithin=section]{theorem}
\declaretheorem[style=theoremstyle, numberwithin=section]{proposition}
\declaretheorem[style=theoremstyle, numberwithin=section]{lemma}
\declaretheorem[style=theoremstyle, numberwithin=section]{corollary}
\declaretheorem[style=theoremstyle, numberwithin=section]{example}

\let\oldproof\proof
\let\endoldproof\endproof

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% caption options
\captionsetup[figure]{labelfont={bf},name={Figure},labelsep=period, font=it}
\captionsetup[table]{labelfont={bf},name={Table},labelsep=period, font=it}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lof/lot/loa/loe options
\newcommand{\addtextlist}[2]{%
\addtocontents{#1}{\protect\vspace{0.3cm}{\Large\bf #2}\vspace{0.2cm}\par}}
\newcommand{\addchapterlof}{\addtextlist{lof}{Chapter \thechapter}}
\newcommand{\addchapterlot}{\addtextlist{lot}{Chapter \thechapter}}
\newcommand{\addchapterloa}{\addtextlist{loa}{Chapter \thechapter}}
\newcommand{\addchapterloe}{\addtextlist{loe}{Chapter \thechapter}}

% https://stackoverflow.com/questions/2785260/hide-an-entry-from-toc-in-latex
\newcommand{\nocontentsline}[3]{}
\newenvironment{noaddcontents}{%
\let\oldaddcontentsline\addcontentsline%
\let\addcontentsline\nocontentsline}{%
\let\addcontentsline\oldaddcontentsline}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cleveref options
\crefname{equation}{Eq.}{Eqs.}
\Crefname{equation}{Equation}{Equations}

\crefname{theorem}{Th.}{Ths.}
\Crefname{theorem}{Theorem}{Theorems}

\crefname{corollary}{Cor.}{Cors.}
\Crefname{corollary}{Corollary}{Corollaries}

\crefname{algorithm}{Algo.}{Algos.}
\Crefname{algorithm}{Algorithm}{Algorithms}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tabular options
% https://texblog.org/2014/10/24/removinghiding-a-column-in-a-latex-table/
\newcolumntype{H}{>{\setbox0=\hbox\bgroup}c<{\egroup}@{}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tcolorbox options

\newtcolorbox{xbox}[2]{%
    left=0.1cm,
    right=0.1cm,
    tikznode boxed title,
    enhanced,
    colframe={#2},
    arc=0mm,
    boxrule=0.02cm,
    interior style={white},
    attach boxed title to top left= {yshift=-\tcboxedtitleheight/2, xshift=1mm},
    fonttitle=\bfseries,
    colbacktitle=white,coltitle={#2},
    boxed title style={size=normal,boxrule=0.0cm,colframe=white},
    title=\fontsize{11pt}{24}\selectfont{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% https://tex.stackexchange.com/questions/18191/defining-custom-labels
\makeatletter
\newcommand{\customlabel}[2]{%
 \@bsphack\begingroup
 \def\@currentlabel{#2}%
 \label{#1}%
 \endgroup\@esphack
}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The Pokeball command!

\NewDocumentCommand{\pokeball}{}
{%
\begin{tikzpicture}[x=1ex, y=1ex, thick, scale=0.1, every node/.style={scale=0.1}]
\draw[line width=0.001] (0,0) circle (1);
    \begin{scope}
        \clip (0,0) circle (1);
        \draw[line width=0.001] (-1,0) -- (1,0);
        \clip (-1,0) rectangle (1,1);
        \draw[line width=0.001, fill=black] (-1,0) rectangle (1,1);
    \end{scope}
    \draw[line width=0.001] (0,0) circle (1);
    \draw[line width=0.001, fill=white] (0,0) circle (0.25);
    \draw[line width=0.001, fill=black] (0,0) circle (0.05);
\end{tikzpicture}%
}