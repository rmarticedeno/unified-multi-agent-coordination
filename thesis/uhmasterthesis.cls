% ============================================================================
% uhmasterthesis.cls
% Curated Master's thesis class with University of Havana front-page branding.
% Front page adapted from the UH thesis style found in the provided template,
% while keeping a reusable master-thesis structure.
% ============================================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uhmasterthesis}[2026/06/19 UH-branded curated Master's thesis class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass{book}

% -----------------------------
% Layout and typography
% -----------------------------
\RequirePackage[a4paper,margin=2.8cm]{geometry}
\RequirePackage{setspace}
\onehalfspacing
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx} % closer to the original UH front-page look
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage[nottoc,notlof,notlot]{tocbibind}
\RequirePackage{etoolbox}
\RequirePackage{ifthen}
\RequirePackage{qrcode}
\RequirePackage{url}

% -----------------------------
% Metadata commands
% -----------------------------
\newcommand{\university}[1]{\def\@university{#1}}
\newcommand{\advisor}[1]{\def\@advisor{#1}}
\newcommand{\coadvisor}[1]{\def\@coadvisor{#1}}
\newcommand{\degree}[1]{\def\@degree{#1}}
\newcommand{\faculty}[1]{\def\@faculty{#1}}
\newcommand{\worktype}[1]{\def\@worktype{#1}}
\newcommand{\logo}[1]{\def\@logo{#1}}
\newcommand{\qrlogo}[1]{\def\@qrlogo{#1}}
\newcommand{\repositoryurl}[1]{\def\@repositoryurl{#1}}

\university{Universidad de La Habana}
\advisor{}
\coadvisor{}
\degree{Máster en Ciencia de la Computación}
\faculty{Facultad de Matemática y Computación}
\worktype{Tesis de Maestría}
\logo{Graphics/uhlogo}
\qrlogo{}
\repositoryurl{}

% -----------------------------
% Header and footer
% -----------------------------
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\nouppercase{\leftmark}}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{plain}{%
    \fancyhf{}%
    \fancyfoot[C]{\bfseries \thepage}%
    \renewcommand{\headrulewidth}{0pt}%
}

% -----------------------------
% Front/main/back matter
% -----------------------------
\renewcommand{\frontmatter}{%
    \cleardoublepage
    \pagenumbering{roman}
    \pagestyle{plain}
}

\renewcommand{\mainmatter}{%
    \cleardoublepage
    \pagenumbering{arabic}
    \pagestyle{fancy}
}

\renewcommand{\backmatter}{%
    \cleardoublepage
    \pagestyle{plain}
}

% -----------------------------
% Environments
% -----------------------------
\newenvironment{dedication}{%
    \cleardoublepage
    \thispagestyle{empty}
    \vspace*{4cm}
    \begin{flushright}
    \itshape
}{%
    \end{flushright}
    \cleardoublepage
}

\newenvironment{acknowledgements}{%
    \chapter*{Agradecimientos}
    \addcontentsline{toc}{chapter}{Agradecimientos}
}{\cleardoublepage}

\newenvironment{abstractES}{%
    \chapter*{Resumen}
    \addcontentsline{toc}{chapter}{Resumen}
}{\cleardoublepage}

\newenvironment{abstractEN}{%
    \chapter*{Abstract}
    \addcontentsline{toc}{chapter}{Abstract}
}{\cleardoublepage}

\newenvironment{introduction}{%
    \chapter*{Introducción}
    \addcontentsline{toc}{chapter}{Introducción}
    \markboth{Introducción}{}
}{\cleardoublepage}

\newenvironment{conclusions}{%
    \chapter*{Conclusiones}
    \addcontentsline{toc}{chapter}{Conclusiones}
    \markboth{Conclusiones}{}
}{\cleardoublepage}

\newenvironment{recommendations}{%
    \chapter*{Recomendaciones y trabajo futuro}
    \addcontentsline{toc}{chapter}{Recomendaciones y trabajo futuro}
    \markboth{Recomendaciones y trabajo futuro}{}
}{\cleardoublepage}

\newenvironment{thesisglossary}{%
    \chapter*{Glosario}
    \addcontentsline{toc}{chapter}{Glosario}
    \markboth{Glosario}{}
}{\cleardoublepage}

% -----------------------------
% Helper for optional images
% -----------------------------
\newcommand{\includeIfExists}[2]{%
    \IfFileExists{#1.pdf}{\includegraphics[#2]{#1}}{%
    \IfFileExists{#1.png}{\includegraphics[#2]{#1}}{%
    \IfFileExists{#1.jpg}{\includegraphics[#2]{#1}}{}}}%
}

% -----------------------------
% UH-style title page
% -----------------------------
\renewcommand{\maketitle}{%
  \thispagestyle{empty}%
  \begin{center}
    \vspace*{1.7cm}
    {\large \@university\par}
    {\large \@faculty\par}
    \vspace{0.25cm}
    \ifstrempty{\@logo}{}{\includeIfExists{\@logo}{width=1.75cm}\par\vspace{0.15cm}}
    \vfill
    {\LARGE \textbf{\@title}\par}
    \vspace{1.8cm}
    {\Large Autor: \textbf{\@author}\par}
    \ifdefempty{\@advisor}{}{\vspace{0.5cm}{\Large Tutor: \textbf{\@advisor}\par}}
    \ifdefempty{\@coadvisor}{}{\vspace{0.2cm}{\Large Cotutor: \textbf{\@coadvisor}\par}}
    \vfill
    {\large \@worktype\par presentada en opción al título de\par \@degree\par}
    \ifstrempty{\@qrlogo}{}{\vspace{0.25cm}\includeIfExists{\@qrlogo}{width=2.3cm}\par}
    \ifstrempty{\@repositoryurl}{}{%
      \vspace{0.25cm}%
      \qrcode[height=2.3cm]{\@repositoryurl}\par
      \vspace{0.1cm}{\footnotesize Repositorio de la tesis\par}%
    }
    \vspace{0.35cm}
    {\large \@date\par}
  \end{center}
  \cleardoublepage
}

% -----------------------------
% Spanish names
% -----------------------------
\AtBeginDocument{%
  \renewcommand{\listtablename}{Índice de tablas}%
  \renewcommand{\tablename}{Tabla}%
}
